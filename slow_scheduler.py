import itertools
import random
import time

# hardcoded data for factory: 10 jobs, 3 machines, processing times in minutes
# iot sensor delays simulated as random additions (0-5 min per task)
jobs = 10
machines = 3
times = [
    [5, 3, 4],  # job 0
    [2, 6, 1],
    [4, 2, 5],
    [3, 4, 2],
    [1, 5, 3],
    [6, 1, 4],
    [2, 3, 5],
    [4, 5, 1],
    [3, 1, 6],
    [5, 4, 2]   # job 9
]

def find_best_schedule():
    start_time = time.time()
    min_makespan = float('inf')
    best_order = None

    # try every possible order - this is slow!
    for perm in itertools.permutations(range(jobs)):
        # simulate iot delays each time (inefficient, recomputes every perm)
        sensor_delays = [[random.randint(0, 5) for _ in range(machines)] for _ in range(jobs)]

        completion = [0] * jobs * machines  # bad flat list instead of 2D
        machine_free_times = [0] * machines

        for i in range(len(perm)):
            job = perm[i]
            prev_complete = 0
            for m in range(machines):
                # wait for machine or previous task
                start = max(machine_free_times[m], prev_complete)
                process_time = times[job][m] + sensor_delays[job][m]  # add iot delay
                complete = start + process_time
                # store in flat list (messy indexing)
                completion[job * machines + m] = complete
                machine_free_times[m] = complete
                prev_complete = complete

        makespan = max(machine_free_times)
        if makespan < min_makespan:
            min_makespan = makespan
            best_order = perm

    end_time = time.time()
    print("Best order:", best_order)
    print("Min makespan:", min_makespan)
    print("Time taken:", end_time - start_time, "seconds")

find_best_schedule()