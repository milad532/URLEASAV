import numpy as np

def sectors_from_angles(
    angles_deg,        # shape (N,), angles in [-180, 180]
    values,            # shape (N,), value per detection
    distances,         # shape (N,), distance per detection
    include_mask=None, # shape (N,), True to include detection
    agg="first",       # "first" (closest), "max", "sum"
    fill_value=0
):
    angles_deg = np.asarray(angles_deg)
    values = np.asarray(values)
    distances = np.asarray(distances)
    
    if include_mask is not None:
        include_mask = np.asarray(include_mask).astype(bool)
    else:
        include_mask = np.ones_like(angles_deg, dtype=bool)
    
    # Filter
    angles_deg = angles_deg[include_mask]
    values = values[include_mask]
    distances = distances[include_mask]
    
    # Handle empty arrays after filtering
    if len(angles_deg) == 0:
        return np.full(12, fill_value, dtype=values.dtype if len(values) > 0 else float)
    
    # Shift [-180,180] → [0,360)
    angles0_360 = (angles_deg + 180) % 360
    
    # Bin indices (12 sectors of 30°)
    # Handle edge case: 360° should map to bin 0, not bin 12
    bins = (angles0_360 // 30).astype(int)
    bins = np.clip(bins, 0, 11)  # Ensure bins are in [0, 11]
    
    if agg == "sum":
        out = np.full(12, fill_value, dtype=values.dtype)
        np.add.at(out, bins, values)
        return out
        
    elif agg == "max":
        out = np.full(12, fill_value, dtype=float)
        # Use a mask to track which bins have been filled
        filled = np.zeros(12, dtype=bool)
        
        for b, v in zip(bins, values):
            if not filled[b] or v > out[b]:
                out[b] = v
                filled[b] = True
        
        return out
        
    elif agg == "first":  # closest object
        out = np.full(12, fill_value, dtype=values.dtype)
        closest_dist = np.full(12, np.inf)
        
        for b, v, d in zip(bins, values, distances):
            if d < closest_dist[b]:
                closest_dist[b] = d
                out[b] = v
                
        return out
        
    else:
        raise ValueError("agg must be 'first', 'max', or 'sum'.")

# -------------------------
# Example usage
# -------------------------
vehicle_walker_angles = np.array([60,2,4])
vehicle_walker_distances = np.array([8,5,2])
values = np.array([10,1,3])  # could be class-based or scores

# Only include walkers (example)
include_mask = None

print("First (closest per bin):", sectors_from_angles(
    vehicle_walker_angles, values, vehicle_walker_distances, include_mask, agg="first"))
print("Max value per bin:", sectors_from_angles(
    vehicle_walker_angles, values, vehicle_walker_distances, include_mask, agg="max"))
print("Sum per bin:", sectors_from_angles(
    vehicle_walker_angles, values, vehicle_walker_distances, include_mask, agg="sum"))

# Test edge cases
print("\nTesting edge cases:")
print("Empty after filtering:", sectors_from_angles(
    np.array([10, 20]), np.array([1, 2]), np.array([5, 10]), 
    np.array([False, False]), agg="first"))
print("Boundary angle 180°:", sectors_from_angles(
    np.array([180]), np.array([99]), np.array([1]), agg="first"))