from datetime import datetime
import yappi
import atexit


def profiler():
    # End profiling and save the results into file
    def output_profiler_stats_file():
        profile_file_name = f'yappi.{datetime.now().isoformat()}.pstat'
        func_stats = yappi.get_func_stats()
        func_stats.save(profile_file_name, type='pstat')
        yappi.stop()
        yappi.clear_stats()

    yappi.start()
    atexit.register(output_profiler_stats_file)
