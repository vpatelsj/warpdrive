package cache

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/dgraph-io/badger/v4"

	"github.com/warpdrive/warpdrive/pkg/backend"
	"github.com/warpdrive/warpdrive/pkg/config"
	"github.com/warpdrive/warpdrive/pkg/metrics"
)

const (
	// DefaultBlockSize is 4 MB.
	DefaultBlockSize = 4 * 1024 * 1024
	// DefaultMaxSize is 50 GB.
	DefaultMaxSize = 50 * 1024 * 1024 * 1024
	// DefaultEvictInterval is the default eviction check period.
	DefaultEvictInterval = 30 * time.Second
	// DefaultEvictTarget is the fraction of max to shrink to when evicting.
	DefaultEvictTarget = 0.85
)

// Stats tracks cache performance metrics (lock-free).
type Stats struct {
	Hits        atomic.Int64
	Misses      atomic.Int64
	Evictions   atomic.Int64
	BytesCached atomic.Int64
}

// Snapshot returns a copy of current stats.
func (s *Stats) Snapshot() StatsSnapshot {
	return StatsSnapshot{
		Hits:        s.Hits.Load(),
		Misses:      s.Misses.Load(),
		Evictions:   s.Evictions.Load(),
		BytesCached: s.BytesCached.Load(),
	}
}

// StatsSnapshot is a point-in-time stats reading.
type StatsSnapshot struct {
	Hits        int64
	Misses      int64
	Evictions   int64
	BytesCached int64
}

// CacheManager provides block-level caching for remote backend data.
//
// Each file is divided into fixed-size blocks (default 4 MB). On read, blocks
// are fetched from the local cache (NVMe) if present, otherwise fetched from
// the remote backend, written to cache, and served. Singleflight deduplication
// ensures concurrent reads for the same block result in one backend fetch.
type CacheManager struct {
	cfg       config.CacheConfig
	cacheDir  string
	blockSize int
	maxSize   int64

	db       *badger.DB
	registry *backend.Registry
	stats    Stats

	// singleflight: in-progress block fetches, keyed by block key.
	inflight sync.Map // map[string]*call

	// fetchSem limits concurrent backend fetches to prevent backend DoS.
	fetchSem chan struct{}

	// blockPool reuses block-sized byte slices to reduce GC pressure.
	blockPool sync.Pool

	// accessMu protects accessTimes.
	accessMu sync.Mutex
	// accessTimes tracks in-memory last-access times, flushed periodically to Badger.
	accessTimes map[string]time.Time

	readahead *ReadaheadManager
	coalescer *ReadCoalescer

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// call represents an in-progress or completed block fetch.
type call struct {
	wg  sync.WaitGroup
	val []byte
	err error
}

// New creates a CacheManager with the given configuration and backend registry.
func New(cfg config.CacheConfig, registry *backend.Registry) (*CacheManager, error) {
	blockSize := DefaultBlockSize
	if cfg.BlockSize > 0 {
		blockSize = int(cfg.BlockSize)
	}

	maxSize := int64(DefaultMaxSize)
	if cfg.MaxSize > 0 {
		maxSize = cfg.MaxSize
	}

	cacheDir := cfg.Path
	if cacheDir == "" {
		cacheDir = "/tmp/warpdrive-cache"
	}

	// Ensure cache directories exist.
	if err := os.MkdirAll(filepath.Join(cacheDir, "blocks"), 0o755); err != nil {
		return nil, fmt.Errorf("cache: create block dir: %w", err)
	}
	metaDir := filepath.Join(cacheDir, "meta")
	if err := os.MkdirAll(metaDir, 0o755); err != nil {
		return nil, fmt.Errorf("cache: create meta dir: %w", err)
	}

	// Open badger for metadata.
	opts := badger.DefaultOptions(metaDir).
		WithLogger(nil).
		WithValueLogFileSize(32 << 20) // 32 MB value log
	db, err := badger.Open(opts)
	if err != nil {
		return nil, fmt.Errorf("cache: open badger: %w", err)
	}

	maxParallel := cfg.MaxParallelFetch
	if maxParallel <= 0 {
		maxParallel = 16
	}

	readaheadWorkers := 4
	if cfg.ReadaheadBlocks > 4 {
		readaheadWorkers = min(cfg.ReadaheadBlocks, 8)
	}

	ctx, cancel := context.WithCancel(context.Background())

	cm := &CacheManager{
		cfg:       cfg,
		cacheDir:  cacheDir,
		blockSize: blockSize,
		maxSize:   maxSize,
		db:        db,
		registry:  registry,
		ctx:       ctx,
		cancel:    cancel,
		fetchSem:  make(chan struct{}, maxParallel),
		blockPool: sync.Pool{
			New: func() any {
				buf := make([]byte, blockSize)
				return &buf
			},
		},
		accessTimes: make(map[string]time.Time),
	}

	// Start readahead manager.
	cm.readahead = NewReadaheadManager(cm, readaheadWorkers)

	// Create read coalescer for access pattern tracking.
	cm.coalescer = NewReadCoalescer(blockSize)

	// Start eviction loop.
	cm.wg.Add(1)
	go cm.evictionLoop(DefaultEvictInterval)

	// Start access time flush loop.
	cm.wg.Add(1)
	go cm.accessFlushLoop(10 * time.Second)

	return cm, nil
}

// Read reads len(p) bytes starting at offset from the specified backend/path.
// It transparently caches data in fixed-size blocks.
func (cm *CacheManager) Read(ctx context.Context, backendName, path string, p []byte, off int64) (int, error) {
	n, _, err := cm.ReadWithCacheHit(ctx, backendName, path, p, off)
	return n, err
}

// ReadWithCacheHit reads len(p) bytes and returns whether any blocks were cache hits.
// Returns (bytesRead, cacheHit, error). cacheHit is true if all blocks were cached.
func (cm *CacheManager) ReadWithCacheHit(ctx context.Context, backendName, path string, p []byte, off int64) (int, bool, error) {
	be, err := cm.registry.Get(backendName)
	if err != nil {
		return 0, false, err
	}

	// Stat remote to get size (uses file-level cache for ETag validation).
	info, err := cm.statWithCache(ctx, be, backendName, path)
	if err != nil {
		return 0, false, err
	}

	if off >= info.Size {
		return 0, false, io.EOF
	}

	// Clamp read to file size.
	remaining := info.Size - off
	if int64(len(p)) > remaining {
		p = p[:remaining]
	}

	totalRead := 0
	allCacheHits := true
	for totalRead < len(p) {
		fileOff := off + int64(totalRead)
		blockIdx := int(fileOff / int64(cm.blockSize))
		blockOff := int(fileOff % int64(cm.blockSize))

		block, cacheHit, err := cm.getBlockWithHit(ctx, be, backendName, path, blockIdx, info.ETag)
		if err != nil {
			if totalRead > 0 {
				return totalRead, allCacheHits, nil
			}
			return 0, false, err
		}
		if !cacheHit {
			allCacheHits = false
		}

		// Guard against short/partial blocks (e.g. last block of a file).
		if blockOff >= len(block) {
			if totalRead > 0 {
				return totalRead, allCacheHits, nil
			}
			return 0, false, fmt.Errorf("cache: block %d is %d bytes but offset is %d for %s/%s", blockIdx, len(block), blockOff, backendName, path)
		}

		n := copy(p[totalRead:], block[blockOff:])
		totalRead += n

		// Record read in coalescer for access pattern detection.
		pattern := cm.coalescer.RecordRead(backendName, path, fileOff, n)

		// Trigger readahead for sequential access (beginning of a new block).
		// Skip readahead if coalescer detects random access.
		if blockOff == 0 && pattern != patternRandom {
			cm.readahead.TriggerReadahead(backendName, path, blockIdx+1, info.ETag, cm.cfg.ReadaheadBlocks)
		}
	}

	return totalRead, allCacheHits, nil
}

// getBlock returns a cached block or fetches it from the backend.
// Uses singleflight dedup so concurrent requests for the same block
// result in a single backend fetch.
func (cm *CacheManager) getBlock(ctx context.Context, be backend.Backend, backendName, path string, blockIdx int, etag string) ([]byte, error) {
	data, _, err := cm.getBlockWithHit(ctx, be, backendName, path, blockIdx, etag)
	return data, err
}

// getBlockWithHit returns a cached block or fetches it, along with cache hit status.
func (cm *CacheManager) getBlockWithHit(ctx context.Context, be backend.Backend, backendName, path string, blockIdx int, etag string) ([]byte, bool, error) {
	key := blockKey(backendName, path, blockIdx)

	// Try cache first.
	data, err := cm.loadFromCache(key, backendName, path, blockIdx, etag)
	if err == nil {
		cm.stats.Hits.Add(1)
		metrics.CacheHits.Inc()
		return data, true, nil
	}

	cm.stats.Misses.Add(1)
	metrics.CacheMisses.Inc()

	// Singleflight: use a pre-initialized call with wg count=1 so
	// waiters always block until the leader finishes.
	newCall := &call{}
	newCall.wg.Add(1)

	if c, loaded := cm.inflight.LoadOrStore(key, newCall); loaded {
		// Another goroutine is already fetching; wait for it.
		existing := c.(*call)
		existing.wg.Wait()
		return existing.val, false, existing.err
	}

	// We are the leader; fetch from backend.
	// Use defer to guarantee wg.Done + inflight cleanup even on panic.
	defer func() {
		if r := recover(); r != nil {
			newCall.err = fmt.Errorf("cache: panic fetching block %d of %s/%s: %v", blockIdx, backendName, path, r)
			slog.Error("panic in block fetch", "block", blockIdx, "backend", backendName, "path", path, "panic", r)
		}
		newCall.wg.Done()
		cm.inflight.Delete(key)
	}()

	data, err = cm.fetchBlock(ctx, be, backendName, path, blockIdx, etag)
	newCall.val = data
	newCall.err = err

	return data, false, err
}

// fetchBlock reads a block from the remote backend and stores it in cache.
func (cm *CacheManager) fetchBlock(ctx context.Context, be backend.Backend, backendName, path string, blockIdx int, etag string) ([]byte, error) {
	// Acquire semaphore to limit concurrent backend fetches.
	select {
	case cm.fetchSem <- struct{}{}:
		defer func() { <-cm.fetchSem }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	offset := int64(blockIdx) * int64(cm.blockSize)

	// Get buffer from pool to reduce GC pressure.
	bufPtr := cm.blockPool.Get().(*[]byte)
	buf := (*bufPtr)[:cm.blockSize]

	n, err := be.ReadAt(ctx, path, buf, offset)
	if err != nil && !errors.Is(err, io.EOF) {
		cm.blockPool.Put(bufPtr)
		return nil, fmt.Errorf("cache: fetch block %d of %s/%s: %w", blockIdx, backendName, path, err)
	}

	// Empty read at EOF — don't cache a zero-length block.
	if n == 0 {
		cm.blockPool.Put(bufPtr)
		return nil, io.EOF
	}

	// Copy result out of pooled buffer.
	result := make([]byte, n)
	copy(result, buf[:n])
	cm.blockPool.Put(bufPtr)

	// Write to local NVMe (atomic: write temp + rename to avoid partial reads).
	localPath := blockLocalPath(cm.cacheDir, backendName, path, blockIdx)
	if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
		return result, nil // serve from memory even if cache write fails
	}
	tmpPath := localPath + ".tmp"
	if err := os.WriteFile(tmpPath, result, 0o644); err != nil {
		os.Remove(tmpPath)
		return result, nil
	}
	if err := os.Rename(tmpPath, localPath); err != nil {
		os.Remove(tmpPath)
		return result, nil
	}

	// Store metadata in badger.
	meta := BlockMeta{
		BackendName: backendName,
		Path:        path,
		BlockIndex:  blockIdx,
		LocalPath:   localPath,
		Size:        n,
		ETag:        etag,
		CachedAt:    time.Now(),
		LastAccess:  time.Now(),
	}
	if err := cm.putBlockMeta(blockKey(backendName, path, blockIdx), &meta); err != nil {
		// Non-fatal; block is still in memory.
		return result, nil
	}

	cm.stats.BytesCached.Add(int64(n))
	metrics.CacheSize.Set(float64(cm.stats.BytesCached.Load()))
	if cm.maxSize > 0 {
		metrics.CacheUtilization.Set(float64(cm.stats.BytesCached.Load()) / float64(cm.maxSize))
	}
	return result, nil
}

// loadFromCache reads a block from local cache if it exists and ETag matches.
func (cm *CacheManager) loadFromCache(key, backendName, path string, blockIdx int, etag string) ([]byte, error) {
	meta, err := cm.getBlockMeta(key)
	if err != nil {
		return nil, err
	}

	// ETag mismatch → stale cache.
	if etag != "" && meta.ETag != etag {
		cm.deleteBlock(key, meta)
		return nil, fmt.Errorf("cache: etag mismatch")
	}

	data, err := os.ReadFile(meta.LocalPath)
	if err != nil {
		// File missing from disk → clean up metadata.
		cm.deleteBlock(key, meta)
		return nil, err
	}

	// Update last access time in memory (avoid Badger write on every hit).
	cm.accessMu.Lock()
	cm.accessTimes[key] = time.Now()
	cm.accessMu.Unlock()

	return data, nil
}

// statWithCache caches file-level stat info to avoid repeated remote calls.
func (cm *CacheManager) statWithCache(ctx context.Context, be backend.Backend, backendName, path string) (backend.ObjectInfo, error) {
	fk := fileKey(backendName, path)

	// Check cached file meta.
	var fm FileMeta
	err := cm.db.View(func(txn *badger.Txn) error {
		item, err := txn.Get([]byte(fk))
		if err != nil {
			return err
		}
		return item.Value(func(val []byte) error {
			return json.Unmarshal(val, &fm)
		})
	})

	// If cached and validated within StaleTTL, return cached version.
	staleTTL := cm.cfg.StaleTTL
	if staleTTL == 0 {
		staleTTL = 30 * time.Second
	}
	if err == nil && time.Since(fm.LastValidated) < staleTTL {
		return backend.ObjectInfo{
			Path: path,
			Size: fm.Size,
			ETag: fm.ETag,
		}, nil
	}

	// Fetch from remote.
	info, err := be.Stat(ctx, path)
	if err != nil {
		return backend.ObjectInfo{}, err
	}

	// Update file meta cache.
	fm = FileMeta{
		BackendName:   backendName,
		Path:          path,
		Size:          info.Size,
		ETag:          info.ETag,
		LastValidated: time.Now(),
	}
	data, _ := json.Marshal(fm)
	_ = cm.db.Update(func(txn *badger.Txn) error {
		return txn.Set([]byte(fk), data)
	})

	return info, nil
}

// deleteBlock removes a block from cache (both disk and metadata).
func (cm *CacheManager) deleteBlock(key string, meta *BlockMeta) {
	_ = os.Remove(meta.LocalPath)
	_ = cm.db.Update(func(txn *badger.Txn) error {
		return txn.Delete([]byte(key))
	})
	cm.stats.BytesCached.Add(-int64(meta.Size))
	cm.stats.Evictions.Add(1)
}

// getBlockMeta retrieves block metadata from badger.
func (cm *CacheManager) getBlockMeta(key string) (*BlockMeta, error) {
	var meta BlockMeta
	err := cm.db.View(func(txn *badger.Txn) error {
		item, err := txn.Get([]byte(key))
		if err != nil {
			return err
		}
		return item.Value(func(val []byte) error {
			return json.Unmarshal(val, &meta)
		})
	})
	if err != nil {
		return nil, err
	}
	return &meta, nil
}

// putBlockMeta stores block metadata in badger.
func (cm *CacheManager) putBlockMeta(key string, meta *BlockMeta) error {
	data, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	return cm.db.Update(func(txn *badger.Txn) error {
		return txn.Set([]byte(key), data)
	})
}

// SetDirFiles informs the readahead manager about the files in a directory.
// This enables directory-level prefetch when files are read in sorted order.
func (cm *CacheManager) SetDirFiles(backendName, dirPath string, files []string) {
	if cm.readahead != nil {
		cm.readahead.SetDirFiles(backendName, dirPath, files)
	}
}

// Stats returns current cache statistics.
func (cm *CacheManager) GetStats() StatsSnapshot {
	return cm.stats.Snapshot()
}

// Close shuts down the cache manager gracefully.
func (cm *CacheManager) Close() error {
	cm.cancel()
	cm.readahead.Stop()
	cm.wg.Wait()
	cm.flushAccessTimes()
	return cm.db.Close()
}

// accessFlushLoop periodically flushes in-memory access times to Badger.
func (cm *CacheManager) accessFlushLoop(interval time.Duration) {
	defer cm.wg.Done()

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-cm.ctx.Done():
			return
		case <-ticker.C:
			cm.flushAccessTimes()
		}
	}
}

// flushAccessTimes writes batched access times to Badger.
func (cm *CacheManager) flushAccessTimes() {
	cm.accessMu.Lock()
	batch := cm.accessTimes
	cm.accessTimes = make(map[string]time.Time, len(batch))
	cm.accessMu.Unlock()

	if len(batch) == 0 {
		return
	}

	for key, t := range batch {
		meta, err := cm.getBlockMeta(key)
		if err != nil {
			continue
		}
		meta.LastAccess = t
		_ = cm.putBlockMeta(key, meta)
	}
}

// getAccessTime returns the most recent access time for a block,
// checking in-memory cache first, falling back to the stored metadata.
func (cm *CacheManager) getAccessTime(key string, storedTime time.Time) time.Time {
	cm.accessMu.Lock()
	t, ok := cm.accessTimes[key]
	cm.accessMu.Unlock()
	if ok && t.After(storedTime) {
		return t
	}
	return storedTime
}
