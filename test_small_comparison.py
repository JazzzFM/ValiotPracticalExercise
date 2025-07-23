#!/usr/bin/env python3
"""Test comparison with small job sets where brute force is feasible."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.job import Job
from src.factories.scheduler_factory import create_scheduler
import time

def test_small_comparison():
    """Compare strategies on small problem where brute force is feasible."""
    
    # Create small job set (6 jobs)
    small_jobs = [
        Job(id=0, processing_times=[5, 3, 4]),
        Job(id=1, processing_times=[2, 6, 1]),
        Job(id=2, processing_times=[4, 2, 5]),
        Job(id=3, processing_times=[3, 4, 2]),
        Job(id=4, processing_times=[1, 5, 3]),
        Job(id=5, processing_times=[6, 1, 4])
    ]
    
    num_machines = 3
    
    print("SMALL PROBLEM COMPARISON (6 jobs, 3 machines)")
    print("="*60)
    
    strategies = [
        ("brute_force", "Brute Force"),
        ("optimized_balanced", "Optimized Balanced"),
        ("optimized_shortest_processing_time", "Shortest Processing Time"),
        ("optimized_random_search", "Random Search")
    ]
    
    results = {}
    
    for strategy_name, display_name in strategies:
        print(f"\nTesting {display_name}...")
        
        try:
            scheduler = create_scheduler(strategy_name)
            
            start_time = time.time()
            result = scheduler.find_optimal_schedule(
                small_jobs, num_machines, 
                use_fixed_delays=True,  # For deterministic results
                max_iterations=500      # Limit for random search
            )
            execution_time = time.time() - start_time
            
            results[strategy_name] = {
                'display_name': display_name,
                'makespan': result.makespan,
                'sequence': result.job_sequence,
                'execution_time': execution_time,
                'utilization': result.average_utilization
            }
            
            print(f"  Makespan: {result.makespan}")
            print(f"  Time: {execution_time:.4f}s")
            print(f"  Sequence: {result.job_sequence}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary comparison
    print(f"\n{'COMPARISON SUMMARY'}")
    print("="*60)
    print(f"{'Strategy':<25} {'Time (s)':<10} {'Makespan':<10} {'Quality':<10}")
    print("-"*60)
    
    best_makespan = min(r['makespan'] for r in results.values() if 'makespan' in r)
    
    for strategy_name, data in results.items():
        if 'makespan' in data:
            quality = data['makespan'] / best_makespan
            print(f"{data['display_name']:<25} {data['execution_time']:<10.4f} "
                  f"{data['makespan']:<10} {quality:<10.2f}x")
    
    # Performance insights
    fastest = min(results.values(), key=lambda x: x.get('execution_time', float('inf')))
    best_quality = min(results.values(), key=lambda x: x.get('makespan', float('inf')))
    
    print(f"\nPERFORMANCE INSIGHTS:")
    print(f"  Fastest: {fastest['display_name']} ({fastest['execution_time']:.4f}s)")
    print(f"  Best Quality: {best_quality['display_name']} (makespan {best_quality['makespan']})")
    
    if fastest != best_quality:
        speedup = fastest['execution_time'] / best_quality['execution_time']
        print(f"  Trade-off: {speedup:.1f}x speed difference between fastest and best quality")

if __name__ == "__main__":
    test_small_comparison()