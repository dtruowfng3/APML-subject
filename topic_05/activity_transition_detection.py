import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def load_har_data():
    """Load toàn bộ dữ liệu HAR từ UCI dataset"""
    base_dir = "C:/Users/TRUONGVO/OneDrive/Desktop/APML/topic_05/archive/UCI HAR Dataset"
    
    def load_data(X_path, y_path):
        X = pd.read_csv(X_path, sep=r"\s+", header=None)
        y = pd.read_csv(y_path, sep=r"\s+", header=None)[0]
        return X, y

    X_train_path = f"{base_dir}/train/X_train.txt"
    y_train_path = f"{base_dir}/train/y_train.txt"
    X_test_path  = f"{base_dir}/test/X_test.txt"
    y_test_path  = f"{base_dir}/test/y_test.txt"
    activity_labels_path = f"{base_dir}/activity_labels.txt"
    features_path = f"{base_dir}/features.txt"

    X_train, y_train = load_data(X_train_path, y_train_path)
    X_test, y_test = load_data(X_test_path, y_test_path)

    activity_labels = pd.read_csv(activity_labels_path, sep=r'\s+', header=None, names=['id', 'label'])
    features = pd.read_csv(features_path, sep=r'\s+', header=None, names=['id', 'name'])

    X_train.columns = X_test.columns = features['name'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'activity_labels': activity_labels,
        'features': features,
        'scaler': scaler
    }

def find_optimal_threshold(sensor_data, window_size=30):
    """Tính difference giữa các cửa sổ để chọn threshold"""
    differences = []
    
    for i in range(window_size, len(sensor_data) - window_size):
        prev_window = sensor_data[i-window_size:i]
        next_window = sensor_data[i:i+window_size]
        
        prev_mean = np.mean(prev_window, axis=0)
        next_mean = np.mean(next_window, axis=0)
        difference = np.linalg.norm(prev_mean - next_mean)
        differences.append(difference)
    
    differences = np.array(differences)

    mean_diff = np.mean(differences)
    std_diff = np.std(differences)

    optimal_threshold = mean_diff + 2 * std_diff
    
    print(f"Difference statistics: mean={mean_diff:.3f}, std={std_diff:.3f}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    
    return optimal_threshold, differences

def improved_transition_detection(sensor_data, window_size=30, threshold_scale=0.1):
    """Phát hiện điểm chuyển đổi dựa trên threshold tối ưu"""
    optimal_threshold, differences = find_optimal_threshold(sensor_data, window_size)
    threshold = optimal_threshold * threshold_scale

    peaks, _ = find_peaks(differences, height=threshold, distance=window_size//2)
    
    transition_points = [p + window_size for p in peaks]
    
    print(f"Using threshold: {threshold:.3f}")
    print(f"Found {len(peaks)} peaks above threshold")
    
    return transition_points, differences

def detect_from_labels(labels):
    transitions = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            transitions.append(i)
    return transitions

def evaluate_transitions(true_labels, detected_points, tolerance=20):
    true_transitions = detect_from_labels(true_labels)
    
    true_positives = 0
    matched_detected = set()  # tránh đếm 1 detected point nhiều lần
    for true_point in true_transitions:
        for idx, detected_point in enumerate(detected_points):
            if idx in matched_detected:
                continue
            if abs(true_point - detected_point) <= tolerance:
                true_positives += 1
                matched_detected.add(idx)
                break
    
    precision = true_positives / len(detected_points) if len(detected_points) > 0 else 0
    recall = true_positives / len(true_transitions) if len(true_transitions) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_transitions': len(true_transitions),
        'detected_transitions': len(detected_points),
        'true_positives': true_positives
    }


def plot_detection_analysis(sensor_data, labels, detected_points, differences, 
                           activity_names, start_idx=0, end_idx=500):
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    time_axis = np.arange(start_idx, end_idx)
    axes[0].plot(time_axis, sensor_data[start_idx:end_idx, 0], 'b-', alpha=0.7)

    for j in range(start_idx + 1, end_idx):
        if labels[j] != labels[j-1]:
            axes[0].axvline(x=j, color='green', linestyle='-', alpha=0.5, label='Actual' if j == start_idx+1 else "")

    for point in detected_points:
        if start_idx <= point < end_idx:
            axes[0].axvline(x=point, color='red', linestyle='--', alpha=0.8, label='Detected' if point == detected_points[0] else "")
    
    axes[0].set_ylabel('Sensor Data')
    axes[0].legend()
    axes[0].set_title('Activity Transition Detection')

    diff_start = max(0, start_idx - 30)
    diff_end = min(len(differences), end_idx - 30)
    diff_axis = np.arange(diff_start + 30, diff_end + 30)
    
    axes[1].plot(diff_axis, differences[diff_start:diff_end], 'orange', alpha=0.7)
    axes[1].axhline(y=np.mean(differences) + 2*np.std(differences), color='red', 
                   linestyle='--', label='Threshold')
    axes[1].set_ylabel('Difference Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, labels[start_idx:end_idx], 'purple', linewidth=2)
    axes[2].set_ylabel('Activity')
    axes[2].set_xlabel('Time Samples')
    axes[2].set_yticks(range(1, 7))
    axes[2].set_yticklabels(activity_names)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_optimized_analysis():
    print("=== OPTIMIZED HAR ACTIVITY TRANSITION DETECTION ===")

    print("1. Loading data...")
    data = load_har_data()
 
    sample_size = 1000
    X_sample = data['X_test_scaled'][:sample_size]
    y_sample = data['y_test'].values[:sample_size]
    activity_names = data['activity_labels']['label'].values
    
    print(f"   Samples: {sample_size}")
    print(f"   Activities: {list(activity_names)}")

    print("\n2. Finding transitions with optimal threshold...")
    detected_points, differences = improved_transition_detection(
        X_sample, 
        window_size=40,
        threshold_scale=0.05
    )
    
    true_points = detect_from_labels(y_sample)
    print(f"   True transitions: {len(true_points)}")
    print(f"   Detected transitions: {len(detected_points)}")

    print("\n3. Evaluation results:")
    metrics = evaluate_transitions(y_sample, detected_points)
    
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall:    {metrics['recall']:.3f}")
    print(f"   F1 Score:  {metrics['f1_score']:.3f}")
    print(f"   True Positives: {metrics['true_positives']}/{metrics['true_transitions']}")

    print("\n4. Sample transitions:")
    for i, point in enumerate(true_points[:5]):
        if point < len(y_sample):
            from_act = activity_names[y_sample[point-1]-1]
            to_act = activity_names[y_sample[point]-1]
            print(f"   {point}: {from_act} → {to_act}")

    print("\n5. Generating detailed analysis...")
    plot_detection_analysis(
        X_sample, y_sample, detected_points, differences, activity_names,
        start_idx=200, end_idx=600
    )
    
    return true_points, detected_points, metrics, differences

if __name__ == "__main__":
    print("Starting Optimized Activity Transition Detection...")
    
    try:
        true_trans, detected_trans, metrics, diffs = run_optimized_analysis()
        print("\n" + "="*60)
        print("OPTIMIZED ANALYSIS COMPLETED!")
        print("="*60)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
