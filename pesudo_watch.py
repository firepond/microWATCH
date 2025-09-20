# Simplified Pseudo-code for microWATCH

Initialize: stable_dist = [], change_points = []

for new_batch in data_stream:

    # Lightweight Distance Metric (Optimization 1)
    distance = euclidean_dist(mean(new_batch), mean(stable_dist))

    if distance > threshold:
        change_points.append(current_time)
        # Reset distribution
        stable_dist = [new_batch]
    else:
        # Fixed-Memory Update (Optimization 2)
        update_running_sum(stable_dist, new_batch)