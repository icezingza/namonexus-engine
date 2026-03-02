import cProfile
import pstats
import io
from run_patent_benchmark import run_patent_stability_benchmark

def profile_it():
    pr = cProfile.Profile()
    pr.enable()
    
    # Run the benchmark
    run_patent_stability_benchmark()
    
    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    
    with open("profiling_results.txt", "w", encoding="utf-8") as f:
        f.write(s.getvalue())
    
    print("\n[PROFILING] Results saved to profiling_results.txt")

if __name__ == "__main__":
    profile_it()
