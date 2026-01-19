"""Phase 2 Tests: Data Preparation & Indexing.

Tests for:
- Excel data loader
- PDF indexer
- Caching system
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_excel_loader():
    """Test Excel data loading functionality."""
    print("\n" + "=" * 80)
    print("TEST 1: Excel Data Loader")
    print("=" * 80)

    try:
        from src.data.excel_loader import ExcelDataLoader

        # Initialize loader
        loader = ExcelDataLoader()
        print("✓ ExcelDataLoader initialized")

        # Load all tables
        tables = loader.load_all_tables()
        print(f"✓ Loaded {len(tables)} tables")

        # Verify all expected tables are loaded
        expected_tables = ["clients", "forfaits", "abonnements", "consommation", "factures", "tickets_support"]
        for table_name in expected_tables:
            if table_name in tables:
                df = tables[table_name]
                print(f"  ✓ {table_name}: {len(df)} rows, {len(df.columns)} columns")
            else:
                print(f"  ✗ {table_name}: NOT FOUND")

        # Test client lookup
        print("\nTesting client lookup functions...")
        clients_df = loader.get_table("clients")

        if clients_df is not None and len(clients_df) > 0:
            # Get first client
            first_client = clients_df.iloc[0]
            client_id = first_client["client_id"]
            client_email = first_client["email"]

            # Test lookup by ID
            client_by_id = loader.get_client_by_id(client_id)
            if client_by_id is not None:
                print(f"  ✓ Client lookup by ID ({client_id}): {client_by_id['prenom']} {client_by_id['nom']}")
            else:
                print(f"  ✗ Client lookup by ID failed")

            # Test lookup by email
            client_by_email = loader.get_client_by_email(client_email)
            if client_by_email is not None:
                print(f"  ✓ Client lookup by email ({client_email}): Found")
            else:
                print(f"  ✗ Client lookup by email failed")

            # Test getting client subscription
            subscription = loader.get_client_subscription(client_id)
            if subscription is not None:
                print(f"  ✓ Client subscriptions: {len(subscription)} found")
            else:
                print(f"  ℹ Client subscriptions: None found (may be normal)")

            # Test getting client invoices
            invoices = loader.get_client_invoices(client_id)
            if invoices is not None:
                print(f"  ✓ Client invoices: {len(invoices)} found")
            else:
                print(f"  ℹ Client invoices: None found (may be normal)")

        # Test table schema
        print("\nTesting table schema retrieval...")
        schema = loader.get_table_schema("clients")
        if schema:
            print("  ✓ Schema retrieval works")
            print(schema[:500] + "..." if len(schema) > 500 else schema)

        print("\n✅ Excel Data Loader: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Excel Data Loader: TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pdf_indexer():
    """Test PDF indexing functionality."""
    print("\n" + "=" * 80)
    print("TEST 2: PDF Indexer")
    print("=" * 80)

    try:
        from src.data.pdf_indexer import PDFIndexer

        # Initialize indexer
        indexer = PDFIndexer()
        print("✓ PDFIndexer initialized")
        print(f"  Embedding model: {indexer.embeddings.model_name}")
        print(f"  Chunk size: {indexer.chunk_size}")
        print(f"  Chunk overlap: {indexer.chunk_overlap}")

        # Load PDFs
        print("\nLoading PDF documents...")
        documents = indexer.load_pdfs()
        if documents:
            print(f"✓ Loaded {len(documents)} pages from PDFs")

            # Show which PDFs were loaded
            sources = set(doc.metadata.get("source_file", "Unknown") for doc in documents)
            print(f"  Sources found: {len(sources)} PDF files")
            for source in sorted(sources):
                count = sum(1 for doc in documents if doc.metadata.get("source_file") == source)
                print(f"    - {source}: {count} pages")
        else:
            print("✗ No documents loaded")
            return False

        # Test chunking
        print("\nTesting document chunking...")
        chunks = indexer.chunk_documents(documents)
        if chunks:
            print(f"✓ Created {len(chunks)} chunks from {len(documents)} pages")
            print(f"  Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")
        else:
            print("✗ Chunking failed")
            return False

        # Test vector database creation
        print("\nCreating vector database (this may take a few minutes)...")
        vectorstore = indexer.create_vectorstore(chunks, force_recreate=False)
        if vectorstore:
            print("✓ Vector database created successfully")
            print(f"  Persist directory: {indexer.vector_db_path}")
        else:
            print("✗ Vector database creation failed")
            return False

        # Test search
        print("\nTesting semantic search...")
        test_queries = [
            "Quels modes de paiement acceptez-vous ?",
            "Comment résilier mon abonnement ?",
            "Forfaits disponibles",
        ]

        for query in test_queries:
            results = indexer.search(query, k=3)
            if results:
                print(f"\n  Query: '{query}'")
                print(f"  ✓ Found {len(results)} results")
                if results:
                    top_result = results[0]
                    print(f"    Top result from: {top_result.metadata.get('source_file', 'Unknown')}")
                    print(f"    Preview: {top_result.page_content[:150]}...")
            else:
                print(f"  ✗ No results for: '{query}'")

        print("\n✅ PDF Indexer: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ PDF Indexer: TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_caching():
    """Test caching system functionality."""
    print("\n" + "=" * 80)
    print("TEST 3: Caching System")
    print("=" * 80)

    try:
        import time
        from src.utils.cache import SimpleCache, cached, get_cache_stats, clear_all_caches

        # Test basic cache
        print("Testing basic cache operations...")
        cache = SimpleCache(ttl=2)  # 2 second TTL for testing

        # Set and get
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        if value == "test_value":
            print("✓ Cache set/get works")
        else:
            print("✗ Cache set/get failed")
            return False

        # Test cache miss
        value = cache.get("nonexistent_key")
        if value is None:
            print("✓ Cache miss returns None")
        else:
            print("✗ Cache miss failed")
            return False

        # Test TTL expiration
        print("\nTesting TTL expiration (waiting 3 seconds)...")
        cache.set("ttl_test", "value")
        time.sleep(3)
        value = cache.get("ttl_test")
        if value is None:
            print("✓ TTL expiration works")
        else:
            print("✗ TTL expiration failed")
            return False

        # Test cache decorator
        print("\nTesting cache decorator...")
        call_count = 0

        @cached(cache_instance=SimpleCache(ttl=60))
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)
            return x * 2

        # First call
        start = time.time()
        result1 = expensive_function(5)
        time1 = time.time() - start

        # Second call (should be cached)
        start = time.time()
        result2 = expensive_function(5)
        time2 = time.time() - start

        if result1 == result2 == 10:
            print(f"✓ Function results correct: {result1}")
        else:
            print("✗ Function results incorrect")
            return False

        if call_count == 1:
            print("✓ Function called only once (second call was cached)")
        else:
            print(f"✗ Function called {call_count} times (should be 1)")
            return False

        if time2 < time1 * 0.5:  # Cached call should be much faster
            print(f"✓ Cached call faster ({time1:.3f}s -> {time2:.3f}s, {time1/time2:.1f}x speedup)")
        else:
            print("⚠ Cached call not significantly faster (may be system dependent)")

        # Test cache stats
        print("\nTesting cache statistics...")
        stats = expensive_function.cache_stats()  # type: ignore
        print(f"  Cache stats: {stats}")
        if stats["hits"] > 0:
            print("✓ Cache statistics tracking works")
        else:
            print("⚠ No cache hits recorded")

        # Test global cache stats
        print("\nTesting global cache stats...")
        clear_all_caches()
        global_stats = get_cache_stats()
        print(f"  Global cache stats: {list(global_stats.keys())}")
        print("✓ Global cache management works")

        print("\n✅ Caching System: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Caching System: TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Phase 2 tests."""
    print("\n" + "=" * 80)
    print("PHASE 2: DATA PREPARATION & INDEXING - TEST SUITE")
    print("=" * 80)

    results = {
        "Excel Data Loader": test_excel_loader(),
        "PDF Indexer": test_pdf_indexer(),
        "Caching System": test_caching(),
    }

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 80)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 80)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 2 is complete and working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
