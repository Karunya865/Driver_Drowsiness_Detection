# advanced_detector.py
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import os
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Keep the same model classes as before
class EfficientNetDetector(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(EfficientNetDetector, self).__init__()
        self.model = models.efficientnet_b0(pretrained=pretrained)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.features[-1].parameters():
            param.requires_grad = True
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class ResNetDetector(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNetDetector, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class VGGDetector(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(VGGDetector, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.features[:24].parameters():
            param.requires_grad = True
        for param in self.model.features[24:].parameters():
            param.requires_grad = False
        in_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class AdvancedTemporalAnalyzer:
    """Advanced temporal analysis with multiple techniques"""
    
    def __init__(self, config):
        self.config = config
        
        # Multi-level thresholds
        self.warning_threshold = config.get('WARNING_THRESHOLD', 1.5)
        self.alarm_threshold = config.get('ALARM_THRESHOLD', 3.0)
        self.critical_threshold = config.get('CRITICAL_THRESHOLD', 5.0)
        
        # Sliding window configuration
        self.window_size = config.get('WINDOW_SIZE', 15)  # frames
        self.drowsy_probs = deque(maxlen=self.window_size)
        self.confidence_scores = deque(maxlen=self.window_size)
        
        # Exponential smoothing
        self.smoothing_alpha = config.get('SMOOTHING_ALPHA', 0.7)
        self.smoothed_drowsy_prob = 0.0
        
        # Pattern recognition
        self.pattern_window = config.get('PATTERN_WINDOW', 30)  # frames
        self.drowsy_pattern = deque(maxlen=self.pattern_window)
        
        # Adaptive thresholds
        self.base_alarm_threshold = self.alarm_threshold
        self.adaptive_factor = 1.0
        
        # State variables
        self.current_state = "ALERT"
        self.state_start_time = time.time()
        self.last_alert_time = time.time()
        self.consecutive_alert_frames = 0
        
        # Statistics
        self.drowsy_events = 0
        self.total_drowsy_time = 0.0
        
    def sliding_window_average(self, current_drowsy_prob):
        """Calculate sliding window average of drowsiness probability"""
        self.drowsy_probs.append(current_drowsy_prob)
        
        if len(self.drowsy_probs) < self.window_size // 3:
            # Not enough data yet, use current value
            return current_drowsy_prob
        
        window_avg = np.mean(list(self.drowsy_probs))
        return window_avg
    
    def exponential_smoothing(self, current_drowsy_prob):
        """Apply exponential smoothing to drowsiness probability"""
        if self.smoothed_drowsy_prob == 0.0:  # First frame
            self.smoothed_drowsy_prob = current_drowsy_prob
        else:
            self.smoothed_drowsy_prob = (self.smoothing_alpha * current_drowsy_prob + 
                                       (1 - self.smoothing_alpha) * self.smoothed_drowsy_prob)
        return self.smoothed_drowsy_prob
    
    def pattern_analysis(self, is_drowsy, confidence):
        """Analyze patterns of drowsiness over time"""
        self.drowsy_pattern.append((is_drowsy, confidence))
        
        if len(self.drowsy_pattern) < self.pattern_window:
            return 0.0, "INSUFFICIENT_DATA"
        
        # Calculate pattern metrics
        drowsy_frames = sum(1 for drowsy, _ in self.drowsy_pattern if drowsy)
        drowsy_ratio = drowsy_frames / len(self.drowsy_pattern)
        
        # Detect frequent blinking/yawn patterns
        transitions = 0
        for i in range(1, len(self.drowsy_pattern)):
            if self.drowsy_pattern[i][0] != self.drowsy_pattern[i-1][0]:
                transitions += 1
        
        transition_rate = transitions / len(self.drowsy_pattern)
        
        if drowsy_ratio > 0.7 and transition_rate < 0.1:
            return drowsy_ratio, "PERSISTENT_DROWSY"
        elif transition_rate > 0.3:
            return drowsy_ratio, "FREQUENT_TRANSITIONS"
        else:
            return drowsy_ratio, "NORMAL"
    
    def adaptive_threshold_calibration(self, current_time, confidence, lighting_condition=None):
        """Dynamically adjust thresholds based on context"""
        # Base adaptive factors
        time_since_alert = current_time - self.last_alert_time
        confidence_factor = 0.5 + (confidence * 0.5)  # 0.5-1.0 range
        
        # Time-based adaptation (long driving sessions)
        session_duration = current_time - self.state_start_time
        if session_duration > 3600:  # After 1 hour
            time_factor = 0.8  # Stricter thresholds
        else:
            time_factor = 1.0
        
        # Lighting condition adaptation
        lighting_factor = 1.0
        if lighting_condition == "NIGHT":
            lighting_factor = 0.8  # Stricter at night
        elif lighting_condition == "LOW_LIGHT":
            lighting_factor = 0.9
        
        # High confidence drowsiness -> stricter thresholds
        if confidence > 0.9:
            confidence_factor = 0.7
        
        self.adaptive_factor = confidence_factor * time_factor * lighting_factor
        self.alarm_threshold = self.base_alarm_threshold * self.adaptive_factor
        
        return self.adaptive_factor
    
    def update_state(self, is_drowsy, confidence, current_time, lighting_condition=None):
        """Advanced state machine with multiple temporal techniques"""
        
        # Calculate drowsiness probability (weighted by confidence)
        drowsy_prob = confidence if is_drowsy else (1 - confidence) * 0.1
        
        # Apply multiple temporal smoothing techniques
        window_avg = self.sliding_window_average(drowsy_prob)
        smoothed_prob = self.exponential_smoothing(window_avg)
        
        # Pattern analysis
        pattern_ratio, pattern_type = self.pattern_analysis(is_drowsy, confidence)
        
        # Adaptive threshold calibration
        adaptive_factor = self.adaptive_threshold_calibration(current_time, confidence, lighting_condition)
        
        # Duration calculations
        state_duration = current_time - self.state_start_time
        
        # State transition logic
        new_state = self.current_state
        alarm_level = 0  # 0: None, 1: Warning, 2: Alarm, 3: Critical
        
        if is_drowsy:
            self.consecutive_alert_frames = 0
            
            if pattern_type == "PERSISTENT_DROWSY":
                # Accelerate alarm for persistent patterns
                effective_duration = state_duration * 1.5
            else:
                effective_duration = state_duration
            
            if effective_duration >= self.critical_threshold:
                new_state = "CRITICAL"
                alarm_level = 3
            elif effective_duration >= self.alarm_threshold:
                new_state = "ALARM"
                alarm_level = 2
            elif effective_duration >= self.warning_threshold:
                new_state = "WARNING"
                alarm_level = 1
            else:
                new_state = "DROWSY_DETECTED"
                alarm_level = 0
                
        else:
            self.consecutive_alert_frames += 1
            self.last_alert_time = current_time
            
            # Require sustained alertness to fully recover
            recovery_threshold = min(2.0, self.alarm_threshold * 0.5)
            if self.consecutive_alert_frames >= recovery_threshold * 10:  # Approximate frames
                new_state = "ALERT"
                alarm_level = 0
            else:
                new_state = "RECOVERING"
                alarm_level = 0 if self.current_state == "ALERT" else 1
        
        # State change handling
        if new_state != self.current_state:
            self.state_start_time = current_time
            self.current_state = new_state
            
            if "DROWSY" in new_state or new_state in ["WARNING", "ALARM", "CRITICAL"]:
                self.drowsy_events += 1
        
        # Update statistics
        if is_drowsy:
            self.total_drowsy_time += (1/30)  # Assuming 30 FPS
        
        return {
            'current_state': new_state,
            'previous_state': self.current_state,
            'alarm_level': alarm_level,
            'state_duration': state_duration,
            'smoothed_probability': smoothed_prob,
            'window_average': window_avg,
            'pattern_ratio': pattern_ratio,
            'pattern_type': pattern_type,
            'adaptive_factor': adaptive_factor,
            'effective_threshold': self.alarm_threshold,
            'consecutive_alert_frames': self.consecutive_alert_frames
        }

class AdvancedDrowsinessDetector:
    def __init__(self, model_paths, config, temporal_config=None):
        """
        Advanced drowsiness detector with multiple temporal techniques
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Initialize advanced temporal analyzer
        temporal_config = temporal_config or {}
        self.temporal_analyzer = AdvancedTemporalAnalyzer(temporal_config)
        
        # Load ensemble models (same as before)
        self.models = self._load_models(model_paths)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((config['IMG_SIZE'], config['IMG_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Define class mappings
        self.classes = config['CLASSES']
        self.drowsy_classes = ['Closed_Eyes', 'yawn']
        self.alert_classes = ['Open_Eyes', 'no_yawn']
        
        # Advanced metrics
        self.frame_history = deque(maxlen=100)  # Store last 100 frames analysis
        self.performance_metrics = {
            'total_frames': 0,
            'drowsy_frames': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        print(f"Advanced Drowsiness Detector initialized on {self.device}")

    def _load_models(self, model_paths):
        """Load ensemble models (same as original)"""
        models = []
        model_classes = {
            'efficient': EfficientNetDetector,
            'resnet': ResNetDetector,
            'vgg': VGGDetector
        }
        
        for model_path in model_paths:
            if not model_path:
                continue
                
            model_type = None
            for key in model_classes.keys():
                if key in model_path.lower():
                    model_type = key
                    break
            
            if model_type:
                try:
                    model = model_classes[model_type](
                        num_classes=self.config['NUM_CLASSES'], 
                        pretrained=False
                    )
                    
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model = model.to(self.device)
                    model.eval()
                    models.append(model)
                    print(f"✓ {model_type.upper()} model loaded successfully")
                    
                except Exception as e:
                    print(f"✗ Error loading {model_type}: {e}")
        
        if not models:
            raise ValueError("No models were successfully loaded!")
        
        return models

    def preprocess_frame(self, frame):
        """Preprocess frame (same as original)"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        input_tensor = self.transform(pil_image)
        return input_tensor.unsqueeze(0).to(self.device)

    def ensemble_predict(self, input_batch):
        """Get ensemble prediction (same as original)"""
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(input_batch)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        avg_probs = np.mean(all_probs, axis=0)[0]
        predicted_class = np.argmax(avg_probs)
        confidence = avg_probs[predicted_class]
        
        return predicted_class, confidence, avg_probs

    def is_drowsy(self, predicted_class, confidence):
        """Determine drowsiness with confidence weighting"""
        class_name = self.classes[predicted_class]
        
        # Adaptive confidence threshold based on class
        if class_name == 'Closed_Eyes':
            min_confidence = 0.6  # Higher threshold for closed eyes
        elif class_name == 'yawn':
            min_confidence = 0.55  # Slightly lower for yawning
        else:
            min_confidence = 0.5
        
        if confidence < min_confidence:
            return False, "Low confidence", confidence
        
        if class_name in self.drowsy_classes:
            return True, class_name, confidence
        else:
            return False, class_name, confidence

    def analyze_lighting_conditions(self, frame):
        """Simple lighting condition analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness < 50:
            return "NIGHT"
        elif avg_brightness < 100:
            return "LOW_LIGHT"
        else:
            return "DAYLIGHT"

    def process_frame(self, frame, current_time=None):
        """Advanced frame processing with multiple temporal techniques"""
        if current_time is None:
            current_time = time.time()
        
        # Analyze lighting conditions for adaptive thresholds
        lighting_condition = self.analyze_lighting_conditions(frame)
        
        # Get model predictions
        input_batch = self.preprocess_frame(frame)
        predicted_class, confidence, all_probs = self.ensemble_predict(input_batch)
        is_drowsy, class_name, weighted_confidence = self.is_drowsy(predicted_class, confidence)
        
        # Advanced temporal analysis
        temporal_results = self.temporal_analyzer.update_state(
            is_drowsy=is_drowsy,
            confidence=weighted_confidence,
            current_time=current_time,
            lighting_condition=lighting_condition
        )
        
        # Update performance metrics
        self.performance_metrics['total_frames'] += 1
        if is_drowsy:
            self.performance_metrics['drowsy_frames'] += 1
        
        # Store frame analysis for trend detection
        frame_analysis = {
            'timestamp': current_time,
            'is_drowsy': is_drowsy,
            'class_name': class_name,
            'confidence': confidence,
            'state': temporal_results['current_state'],
            'alarm_level': temporal_results['alarm_level'],
            'lighting': lighting_condition
        }
        self.frame_history.append(frame_analysis)
        
        results = {
            'frame': frame,
            'predicted_class': class_name,
            'confidence': confidence,
            'weighted_confidence': weighted_confidence,
            'is_drowsy': is_drowsy,
            'alarm_level': temporal_results['alarm_level'],
            'current_state': temporal_results['current_state'],
            'state_duration': temporal_results['state_duration'],
            'drowsiness_duration': temporal_results['state_duration'] if is_drowsy else 0,
            'consecutive_frames': self.temporal_analyzer.consecutive_alert_frames,
            'all_probabilities': all_probs,
            'current_time': current_time,
            'temporal_metrics': temporal_results,
            'lighting_condition': lighting_condition,
            'performance_metrics': self.performance_metrics.copy()
        }
        
        return results

    def get_system_health(self):
        """Get system health and performance metrics"""
        total_frames = self.performance_metrics['total_frames']
        drowsy_ratio = (self.performance_metrics['drowsy_frames'] / total_frames) if total_frames > 0 else 0
        
        health_metrics = {
            'total_frames_processed': total_frames,
            'drowsy_frame_ratio': drowsy_ratio,
            'drowsy_events': self.temporal_analyzer.drowsy_events,
            'total_drowsy_time': self.temporal_analyzer.total_drowsy_time,
            'current_adaptive_factor': self.temporal_analyzer.adaptive_factor,
            'system_uptime': time.time() - self.temporal_analyzer.state_start_time,
            'average_confidence': np.mean([f['confidence'] for f in self.frame_history]) if self.frame_history else 0
        }
        
        return health_metrics

    def draw_advanced_detection_info(self, results):
        """Enhanced visualization with advanced metrics"""
        frame = results['frame'].copy()
        height, width = frame.shape[:2]
        
        # Color scheme based on alarm level
        colors = {
            0: (0, 255, 0),    # Green - Alert
            1: (0, 255, 255),  # Yellow - Warning
            2: (0, 165, 255),  # Orange - Alarm
            3: (0, 0, 255)     # Red - Critical
        }
        
        alarm_level = results['alarm_level']
        status_color = colors[alarm_level]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Main status box
        cv2.rectangle(frame, (10, 10), (width-10, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (width-10, 140), status_color, 2)
        
        # Status text based on alarm level
        status_texts = {
            0: "SYSTEM NORMAL",
            1: "WARNING: EARLY DROWSINESS",
            2: "ALARM: DROWSY DRIVING",
            3: "CRITICAL: IMMEDIATE ATTENTION NEEDED"
        }
        
        cv2.putText(frame, status_texts[alarm_level], (20, 35), font, font_scale, status_color, thickness)
        
        # Detailed information
        info_y = 60
        line_height = 20
        
        info_lines = [
            f"State: {results['current_state']} ({results['state_duration']:.1f}s)",
            f"Class: {results['predicted_class']} (conf: {results['confidence']:.2f})",
            f"Lighting: {results['lighting_condition']}",
            f"Pattern: {results['temporal_metrics']['pattern_type']}",
            f"Adaptive Threshold: {results['temporal_metrics']['effective_threshold']:.1f}s"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, info_y + i * line_height), font, font_scale*0.7, (255, 255, 255), 1)
        
        # Probability bars (enhanced)
        self._draw_advanced_probability_bars(frame, results['all_probabilities'], results['temporal_metrics'])
        
        # Temporal analysis graph
        self._draw_temporal_graph(frame, results)
        
        return frame

    def _draw_advanced_probability_bars(self, frame, probabilities, temporal_metrics):
        """Enhanced probability visualization"""
        height, width = frame.shape[:2]
        bar_height = 15
        bar_width = 120
        start_x = width - bar_width - 20
        start_y = 160
        
        for i, (class_name, prob) in enumerate(zip(self.classes, probabilities)):
            y = start_y + i * (bar_height + 8)
            
            # Class label
            cv2.putText(frame, f"{class_name}:", (10, y + bar_height//2 + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Background bar
            cv2.rectangle(frame, (start_x, y), (start_x + bar_width, y + bar_height), 
                         (50, 50, 50), -1)
            
            # Filled bar
            fill_width = int(bar_width * prob)
            color = (0, 255, 0) if prob > 0.7 else (0, 200, 255) if prob > 0.5 else (0, 100, 255)
            cv2.rectangle(frame, (start_x, y), (start_x + fill_width, y + bar_height), 
                         color, -1)
            
            # Probability text
            prob_text = f"{prob:.3f}"
            cv2.putText(frame, prob_text, (start_x + bar_width + 5, y + bar_height//2 + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Smoothed probability indicator
        smoothed_prob = temporal_metrics['smoothed_probability']
        y_smooth = start_y + len(self.classes) * (bar_height + 8) + 10
        cv2.putText(frame, f"Smoothed: {smoothed_prob:.3f}", (start_x, y_smooth), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def _draw_temporal_graph(self, frame, results):
        """Mini temporal analysis graph"""
        height, width = frame.shape[:2]
        graph_width = 200
        graph_height = 80
        graph_x = width - graph_width - 20
        graph_y = height - graph_height - 20
        
        # Graph background
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), 
                     (40, 40, 40), -1)
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), 
                     (100, 100, 100), 1)
        
        # Plot recent drowsiness probabilities
        if len(self.frame_history) > 1:
            recent_frames = list(self.frame_history)[-min(20, len(self.frame_history)):]
            x_vals = np.linspace(graph_x, graph_x + graph_width, len(recent_frames))
            
            for i in range(1, len(recent_frames)):
                y1 = graph_y + graph_height - int(recent_frames[i-1]['confidence'] * graph_height)
                y2 = graph_y + graph_height - int(recent_frames[i]['confidence'] * graph_height)
                x1 = int(x_vals[i-1])
                x2 = int(x_vals[i])
                
                color = (0, 255, 0) if not recent_frames[i]['is_drowsy'] else (0, 0, 255)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        cv2.putText(frame, "Temporal Trend", (graph_x + 5, graph_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def reset_system(self):
        """Reset the entire system state"""
        self.temporal_analyzer = AdvancedTemporalAnalyzer(self.temporal_analyzer.config)
        self.frame_history.clear()
        self.performance_metrics = {
            'total_frames': 0,
            'drowsy_frames': 0,
            'false_positives': 0,
            'false_negatives': 0
        }

# Enhanced main function with advanced features
def advanced_detect_drowsiness_video(video_path, model_paths, config, output_path=None, 
                                   temporal_config=None, display=False, save_output=True):
    """
    Advanced drowsiness detection with multiple temporal techniques
    """
    
    # Enhanced temporal configuration
    default_temporal_config = {
        'WARNING_THRESHOLD': 1.5,
        'ALARM_THRESHOLD': 3.0,
        'CRITICAL_THRESHOLD': 5.0,
        'WINDOW_SIZE': 15,
        'SMOOTHING_ALPHA': 0.7,
        'PATTERN_WINDOW': 30
    }
    
    if temporal_config:
        default_temporal_config.update(temporal_config)
    
    detector = AdvancedDrowsinessDetector(
        model_paths=model_paths,
        config=config,
        temporal_config=default_temporal_config
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Advanced Detection Started:")
    print(f"Video: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    print(f"Temporal Config: {default_temporal_config}")
    
    if save_output:
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"advanced_detection_{base_name}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Enhanced statistics
    statistics = {
        'start_time': time.time(),
        'frame_count': 0,
        'alarm_counts': {0: 0, 1: 0, 2: 0, 3: 0},
        'state_durations': {},
        'pattern_types': {}
    }
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            statistics['frame_count'] += 1
            
            # Process frame with advanced detection
            results = detector.process_frame(frame)
            
            # Update statistics
            alarm_level = results['alarm_level']
            statistics['alarm_counts'][alarm_level] += 1
            
            current_state = results['current_state']
            if current_state not in statistics['state_durations']:
                statistics['state_durations'][current_state] = 0
            statistics['state_durations'][current_state] += (1/fps)
            
            pattern_type = results['temporal_metrics']['pattern_type']
            if pattern_type not in statistics['pattern_types']:
                statistics['pattern_types'][pattern_type] = 0
            statistics['pattern_types'][pattern_type] += 1
            
            # Draw advanced visualization
            annotated_frame = detector.draw_advanced_detection_info(results)
            
            if display:
                cv2.imshow('Advanced Drowsiness Detection', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    detector.reset_system()
                    print("System reset complete")
                elif key == ord('s'):
                    health = detector.get_system_health()
                    print("System Health:", health)
            
            if save_output and out is not None:
                out.write(annotated_frame)
            
            # Progress reporting
            if statistics['frame_count'] % 60 == 0:
                elapsed = time.time() - statistics['start_time']
                fps_processed = statistics['frame_count'] / elapsed
                percent_complete = (statistics['frame_count'] / total_frames) * 100
                
                health = detector.get_system_health()
                print(f"Progress: {percent_complete:.1f}% | FPS: {fps_processed:.1f} | "
                      f"Drowsy: {health['drowsy_frame_ratio']:.1%} | "
                      f"Events: {health['drowsy_events']}")
    
    except KeyboardInterrupt:
        print("Detection interrupted by user")
    except Exception as e:
        print(f"Error during advanced detection: {e}")
    finally:
        cap.release()
        if save_output and out is not None:
            out.release()
        
        if display:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        # Enhanced summary report
        processing_time = time.time() - statistics['start_time']
        health = detector.get_system_health()
        
        print("\n" + "="*60)
        print("ADVANCED DETECTION SUMMARY")
        print("="*60)
        print(f"Total frames: {statistics['frame_count']}")
        print(f"Processing time: {processing_time:.2f}s ({statistics['frame_count']/processing_time:.1f} FPS)")
        print(f"Drowsy ratio: {health['drowsy_frame_ratio']:.1%}")
        print(f"Drowsy events: {health['drowsy_events']}")
        print(f"Total drowsy time: {health['total_drowsy_time']:.1f}s")
        print("\nAlarm Level Distribution:")
        for level, count in statistics['alarm_counts'].items():
            percentage = (count / statistics['frame_count']) * 100
            print(f"  Level {level}: {count} frames ({percentage:.1f}%)")
        print("\nPattern Analysis:")
        for pattern, count in statistics['pattern_types'].items():
            percentage = (count / statistics['frame_count']) * 100
            print(f"  {pattern}: {count} frames ({percentage:.1f}%)")
        
        if output_path and save_output and os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Output saved: {output_path} ({file_size:.1f} MB)")

# Webcam version with advanced features
def advanced_detect_drowsiness_webcam(model_paths, config, temporal_config=None):
    """Advanced real-time webcam detection"""
    
    default_temporal_config = {
        'WARNING_THRESHOLD': 1.0,  # Shorter thresholds for real-time
        'ALARM_THRESHOLD': 2.0,
        'CRITICAL_THRESHOLD': 3.5,
        'WINDOW_SIZE': 10,
        'SMOOTHING_ALPHA': 0.8,    # More responsive in real-time
        'PATTERN_WINDOW': 20
    }
    
    if temporal_config:
        default_temporal_config.update(temporal_config)
    
    detector = AdvancedDrowsinessDetector(
        model_paths=model_paths,
        config=config,
        temporal_config=default_temporal_config
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Advanced Webcam Detection Started")
    print("Controls: 'q'=quit, 'r'=reset, 's'=system health")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = detector.process_frame(frame)
            annotated_frame = detector.draw_advanced_detection_info(results)
            
            cv2.imshow('Advanced Drowsiness Detection - Webcam', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset_system()
                print("System reset complete")
            elif key == ord('s'):
                health = detector.get_system_health()
                print("System Health:", health)
    
    except KeyboardInterrupt:
        print("Webcam detection interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Configuration
    config = {
        'CLASSES': ['Closed_Eyes', 'Open_Eyes', 'yawn', 'no_yawn'],
        'NUM_CLASSES': 4,
        'IMG_SIZE': 224
    }
    
    # Model paths
    model_paths = [
        'saved_models/efficient_best_model.pth',
        'saved_models/resnet_best_model.pth', 
        'saved_models/vgg_best_model.pth'
    ]
    
    # Check models
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✓ Found model: {model_path}")
        else:
            print(f"✗ Missing model: {model_path}")
    
    print("\nAdvanced Drowsiness Detection System")
    print("Choose detection mode:")
    print("1. Advanced video file processing")
    print("2. Advanced real-time webcam")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        video_path = input("Enter video file path: ").strip()
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found!")
        else:
            advanced_detect_drowsiness_video(
                video_path=video_path,
                model_paths=model_paths,
                config=config,
                display=False,
                save_output=True
            )
    
    elif choice == "2":
        advanced_detect_drowsiness_webcam(
            model_paths=model_paths,
            config=config
        )
    
    else:
        print("Invalid choice!")