import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

# Solve OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

# Set English font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class DetectionConfig:
    """Detection configuration parameters"""
    MIN_CONTOUR_AREA: int = 5000
    SOLIDITY_THRESHOLD: float = 0.3
    MORPH_CLOSE_SIZE: Tuple[int, int] = (25, 25)
    MORPH_DILATE_ITERATIONS: int = 2
    MORPH_OPEN_SIZE: Tuple[int, int] = (5, 5)

    # Color ranges
    COLOR_RANGES: List[Tuple[np.ndarray, np.ndarray]] = None

    # Scoring weights
    SCORING_WEIGHTS: Dict[str, float] = None

    # Ideal parameter ranges
    IDEAL_RANGES: Dict[str, Tuple[float, float]] = None

    def __post_init__(self):
        if self.COLOR_RANGES is None:
            self.COLOR_RANGES = [
                (np.array([140, 30, 50]), np.array([180, 255, 255])),  # Pink
                (np.array([100, 30, 50]), np.array([130, 255, 255])),  # Blue
                (np.array([0, 50, 50]), np.array([180, 255, 255]))  # High saturation
            ]

        if self.SCORING_WEIGHTS is None:
            self.SCORING_WEIGHTS = {
                'saturation': 0.25,  # Saturation detection weight
                'color_match': 0.25,  # Color matching
                'area_ratio': 0.40,  # Area ratio
                'solidity': 0.10  # Solidity
            }

        if self.IDEAL_RANGES is None:
            self.IDEAL_RANGES = {
                'area_ratio': (0.15, 1.00),
                'aspect_ratio': (0.6, 2.0),
                'solidity': (0.4, 0.9)
            }


class AdvancedELISADetector:
    """Advanced ELISA reaction region detector from Phase 1"""

    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()

    def create_black_text_mask(self, image):
        """Create black text mask"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 80, 50])
        black_mask = cv2.inRange(hsv, black_lower, black_upper)

        kernel_small = np.ones((2, 2), np.uint8)
        kernel_large = np.ones((5, 5), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_small)
        black_mask = cv2.dilate(black_mask, kernel_large, iterations=1)
        return black_mask

    def remove_black_text_from_image(self, image, black_mask):
        """Remove black text from image"""
        cleaned_image = image.copy()
        black_indices = np.where(black_mask > 0)

        if len(black_indices[0]) > 0:
            kernel_size = 5
            filtered_image = cv2.medianBlur(image, kernel_size)
            cleaned_image[black_indices] = filtered_image[black_indices]

        return cleaned_image

    def calculate_saturation_score(self, cleaned_image, bbox):
        """Calculate saturation score"""
        x, y, w, h = bbox
        roi = cleaned_image[y:y + h, x:x + w]

        if roi.size == 0:
            return 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)

        mean_score = min(sat_mean / 100.0, 1.0)
        std_score = 1.0 - min(abs(sat_std - 40) / 80.0, 1.0)

        saturation_score = 0.7 * mean_score + 0.3 * std_score
        return min(saturation_score, 1.0)

    def calculate_color_match_score(self, candidate_info):
        """Calculate color match score"""
        color_weights = {'High saturation': 1.0, 'Pink': 0.8, 'Blue': 0.6}
        return color_weights.get(candidate_info['color_range'], 0.5)

    def calculate_area_scores(self, candidates):
        """Calculate area scores"""
        ideal_candidates = [c for c in candidates
                            if self.config.IDEAL_RANGES['area_ratio'][0] <= c['area_ratio'] <=
                            self.config.IDEAL_RANGES['area_ratio'][1]]

        if not ideal_candidates:
            ideal_candidates = candidates.copy()

        ideal_candidates.sort(key=lambda x: x['area_ratio'], reverse=True)
        area_scores = {}

        for i, candidate in enumerate(ideal_candidates):
            if i == 0:
                area_weight = 1.0
            elif i == 1:
                area_weight = 0.8
            elif i == 3:
                area_weight = 0.4
            elif i == 4:
                area_weight = 0.2
            else:
                area_weight = 0.1

            area_ratio = candidate['area_ratio']
            ideal_min, ideal_max = self.config.IDEAL_RANGES['area_ratio']

            if area_ratio < ideal_min:
                base_score = area_ratio / ideal_min
            elif area_ratio > ideal_max:
                base_score = max(0, 1.0 - (area_ratio - ideal_max) / (1.0 - ideal_max))
            else:
                base_score = 1.0

            area_scores[candidate['id']] = base_score * area_weight

        for candidate in candidates:
            if candidate['id'] not in area_scores:
                area_scores[candidate['id']] = 0.1

        return area_scores

    def calculate_solidity_score(self, solidity):
        """Calculate solidity score"""
        ideal_min, ideal_max = self.config.IDEAL_RANGES['solidity']
        if solidity < ideal_min:
            return solidity / ideal_min
        elif solidity > ideal_max:
            return max(0, 1.0 - (solidity - ideal_max) / (1.0 - ideal_max))
        else:
            return 1.0

    def enhanced_morphology(self, mask):
        """Enhanced morphological processing"""
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config.MORPH_CLOSE_SIZE)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        dilated_mask = cv2.dilate(closed_mask, kernel_large, iterations=self.config.MORPH_DILATE_ITERATIONS)
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config.MORPH_OPEN_SIZE)
        return cv2.morphologyEx(dilated_mask, cv2.MORPH_OPEN, kernel_medium)

    def detect_all_candidates(self, image):
        """Detect all candidate regions"""
        candidates = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_h, img_w = image.shape[:2]
        image_area = img_h * img_w

        black_text_mask = self.create_black_text_mask(image)
        cleaned_image = self.remove_black_text_from_image(image, black_text_mask)

        for i, (lower, upper) in enumerate(self.config.COLOR_RANGES):
            color_name = ['Pink', 'Blue', 'High saturation'][i]
            mask = cv2.inRange(hsv, lower, upper)
            cleaned_mask = cv2.bitwise_and(mask, cv2.bitwise_not(black_text_mask))
            final_mask = self.enhanced_morphology(cleaned_mask)

            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for j, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < self.config.MIN_CONTOUR_AREA:
                    continue

                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                if solidity < self.config.SOLIDITY_THRESHOLD:
                    continue

                x, y, w, h = cv2.boundingRect(hull)
                bbox_area = w * h
                area_ratio = bbox_area / image_area

                candidate = {
                    'id': f"{color_name}_{j + 1}", 'bbox': (x, y, w, h), 'area': bbox_area,
                    'area_ratio': area_ratio, 'aspect_ratio': w / h, 'solidity': solidity,
                    'center': (x + w / 2, y + h / 2), 'color_range': color_name,
                    'contour': hull, 'contour_area': area, 'hull_area': hull_area
                }
                candidates.append(candidate)

        return candidates, black_text_mask, cleaned_image

    def score_candidates(self, cleaned_image, candidates):
        """Comprehensively score all candidate regions"""
        scored_candidates = []
        area_scores = self.calculate_area_scores(candidates)

        for candidate in candidates:
            saturation_score = self.calculate_saturation_score(cleaned_image, candidate['bbox'])
            color_score = self.calculate_color_match_score(candidate)
            area_score = area_scores.get(candidate['id'], 0.1)
            solidity_score = self.calculate_solidity_score(candidate['solidity'])

            total_score = (
                    saturation_score * self.config.SCORING_WEIGHTS['saturation'] +
                    color_score * self.config.SCORING_WEIGHTS['color_match'] +
                    area_score * self.config.SCORING_WEIGHTS['area_ratio'] +
                    solidity_score * self.config.SCORING_WEIGHTS['solidity']
            )

            scored_candidate = candidate.copy()
            scored_candidate.update({
                'saturation_score': saturation_score, 'color_score': color_score,
                'area_score': area_score, 'solidity_score': solidity_score,
                'total_score': total_score, 'confidence': total_score
            })
            scored_candidates.append(scored_candidate)

        scored_candidates.sort(key=lambda x: x['total_score'], reverse=True)
        return scored_candidates

    def validate_detection_quality(self, detection_result, image_shape):
        """Validate detection quality"""
        if not detection_result:
            return {'is_valid': False, 'message': 'No reaction region detected'}

        img_area = image_shape[0] * image_shape[1]
        region_area = detection_result['area']
        area_ratio = region_area / img_area
        confidence = detection_result['confidence']

        # 评估图片质量
        quality_rating = self.assess_image_quality(confidence)

        quality_issues = []
        if confidence < 0.7:
            quality_issues.append(f"Confidence too low ({confidence:.3f})")
        if area_ratio < 0.05:
            quality_issues.append(f"Detection region too small ({area_ratio:.1%})")
        if area_ratio > 0.8:
            quality_issues.append(f"Detection region too large ({area_ratio:.1%})")

        if quality_issues:
            return {
                'is_valid': False, 'message': 'Poor detection quality', 'issues': quality_issues,
                'metrics': {'confidence': confidence, 'area_ratio': area_ratio,
                            'region_area': region_area, 'image_area': img_area},
                'quality_rating': quality_rating
            }
        else:
            return {
                'is_valid': True, 'message': 'Good detection quality',
                'metrics': {'confidence': confidence, 'area_ratio': area_ratio,
                            'region_area': region_area, 'image_area': img_area},
                'quality_rating': quality_rating
            }

    def assess_image_quality(self, confidence: float) -> Dict[str, str]:
        """
        根据检测置信度评估图片质量

        Args:
            confidence: 检测置信度 (0-1)

        Returns:
            包含质量等级和建议的字典
        """
        if confidence < 0.7:
            return {
                'quality_level': 'Poor',
                'description': 'Image quality: Poor',
                'suggestion': 'Recommendation: Change the angle or re-shoot with different background and upload again.',
                'confidence': confidence
            }
        elif 0.7 <= confidence < 0.9:
            return {
                'quality_level': 'Acceptable',
                'description': 'Image quality: Acceptable',
                'suggestion': 'Image quality is acceptable for analysis.',
                'confidence': confidence
            }
        else:  # confidence >= 0.9
            return {
                'quality_level': 'Excellent',
                'description': 'Image quality: Excellent',
                'suggestion': 'Excellent image quality for precise analysis.',
                'confidence': confidence
            }

    def detect_reaction_region_complete(self, image):
        """Complete reaction region detection"""
        candidates, black_mask, cleaned_image = self.detect_all_candidates(image)
        if not candidates:
            return None, black_mask

        scored_candidates = self.score_candidates(cleaned_image, candidates)
        best_region = scored_candidates[0] if scored_candidates else None

        if best_region:
            quality_check = self.validate_detection_quality(best_region, image.shape)
            best_region['quality_check'] = quality_check

        return best_region, black_mask


class DeepLearningPredictor:
    """Deep learning predictor - UPDATED VERSION"""

    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = ['NEG', 'POS']  # Default class names

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if model_path:
            self.load_model(model_path)

    def create_resnet18_model(self, num_classes=1, dropout_rate=0.3):
        """Create ResNet18 model matching training architecture"""
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        except:
            model = models.resnet18(pretrained=True)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last two blocks
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

        # Modify classifier to match training architecture
        num_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )

        return model

    def load_model(self, model_path):
        """Load trained model - UPDATED for new checkpoint format"""
        if model_path is None or not os.path.exists(model_path):
            print("⚠️  Model file not found, please provide correct model path")
            return None

        try:
            print(f"🔧 Loading classifier model: {model_path}")

            # Try different loading methods
            try:
                # Method 1: Try weights_only=False (PyTorch 2.6+)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            except:
                # Method 2: Try old way
                checkpoint = torch.load(model_path, map_location=self.device)

            # Extract information from checkpoint
            if 'model_state_dict' in checkpoint:
                # New format: contains model_state_dict and class_names
                model_state_dict = checkpoint['model_state_dict']
                self.class_names = checkpoint.get('class_names', ['NEG', 'POS'])
                training_config = checkpoint.get('training_config', {})
                print(f"✅ Checkpoint format: New (contains model_state_dict)")
            else:
                # Old format: directly contains state_dict
                model_state_dict = checkpoint
                print(f"✅ Checkpoint format: Old (direct state_dict)")

            # Create model with correct architecture
            dropout_rate = training_config.get('dropout_rate', 0.3) if 'training_config' in locals() else 0.3
            self.model = self.create_resnet18_model(num_classes=1, dropout_rate=dropout_rate)

            # Load weights
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.model.eval()

            print(f"✅ Model loaded successfully: {model_path}")
            print(f"   Classes: {self.class_names}")
            print(f"   Device: {self.device}")

            return self.model

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_single_image(self, image_array):
        """Predict single image"""
        if self.model is None:
            print("❌ Model not loaded, cannot perform prediction")
            return None, 0.5, 0.5

        try:
            if isinstance(image_array, np.ndarray):
                if image_array.shape[-1] == 3:
                    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
                else:
                    image = Image.fromarray(image_array)
            else:
                image = image_array

            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(image_tensor)
                probability = torch.sigmoid(output).item()

            predicted_class = self.class_names[1] if probability > 0.5 else self.class_names[0]
            confidence = probability if probability > 0.5 else 1 - probability

            return predicted_class, confidence, probability

        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None, 0.5, 0.5


class ELISA_Complete_Analyzer:
    """Complete ELISA analysis system - UPDATED"""

    def __init__(self, grid_rows=4, grid_cols=7, initial_titer=10, high_titer_threshold=160, model_path=None):
        self.high_titer_threshold = high_titer_threshold  # 高抗体效价阈值，默认1:160
        """
        Initialize ELISA analyzer

        Args:
            grid_rows: Number of rows in the grid
            grid_cols: Number of columns in the grid
            initial_titer: Initial antibody titer dilution (e.g., 10 for 1:10)
            model_path: Path to deep learning model
        """
        # Reaction well grid parameters
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.initial_titer = initial_titer  # 初始抗体效价，如10表示1:10

        # Use Phase 1 detector for region detection
        self.region_detector = AdvancedELISADetector()
        self.predictor = DeepLearningPredictor(model_path)

        print(f"🔧 Initialization parameters:")
        print(f"   Grid: {grid_rows}×{grid_cols}")
        print(f"   Initial antibody titer: 1:{initial_titer}")
        print(f"   High titer threshold: 1:{high_titer_threshold}")
        print(f"   Using Phase 1 detection algorithm")
        if model_path and os.path.exists(model_path):
            print(f"🔧 Model path: {model_path}")
        else:
            print("⚠️  Warning: No valid model path provided")

    def preprocess_image(self, image_path, max_size=1200):
        """Image preprocessing"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image_rgb.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h))
            print(f"📐 Image resized: {w}x{h} -> {new_w}x{new_h}")

        return image_rgb

    def create_wells_from_grid(self, image, reaction_region):
        """Create reaction wells based on grid parameters"""
        if reaction_region is None:
            return []

        x, y, w, h = reaction_region['bbox']
        wells = []

        # Calculate size of each reaction well
        well_width = w // self.grid_cols
        well_height = h // self.grid_rows

        print(f"📊 Grid parameters: {self.grid_rows} rows × {self.grid_cols} columns")
        print(f"📏 Well size: {well_width}×{well_height} pixels")

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Calculate well position
                well_x = x + col * well_width
                well_y = y + row * well_height

                # Extract well image
                well_image = image[well_y:well_y + well_height, well_x:well_x + well_width]

                if well_image.size > 0:
                    wells.append({
                        'image': well_image.copy(),
                        'roi': well_image.copy(),  # Add 'roi' key for compatibility
                        'position': (row, col),
                        'global_bbox': (well_x, well_y, well_width, well_height),
                        'grid_index': row * self.grid_cols + col + 1  # Start numbering from 1
                    })

        print(f"🔬 Created {len(wells)} reaction wells")
        return wells

    def analyze_wells_with_dl(self, wells):
        """Analyze each reaction well using deep learning model"""
        if self.predictor.model is None:
            print("❌ Deep learning model not available, skipping prediction")
            # Add default predictions for testing
            for well in wells:
                well['prediction'] = 'NEG'
                well['confidence'] = 0.5
                well['probability'] = 0.5
                well['color'] = 'green'
            return wells

        print("🧠 Starting deep learning prediction...")
        for i, well in enumerate(wells):
            predicted_class, confidence, probability = self.predictor.predict_single_image(well['image'])
            well['prediction'] = predicted_class
            well['confidence'] = confidence
            well['probability'] = probability
            well['color'] = 'red' if predicted_class == 'POS' else 'green'

            if (i + 1) % 10 == 0 or (i + 1) == len(wells):
                print(f"  Prediction progress: {i + 1}/{len(wells)}")

        return wells

    def calculate_antibody_titer(self, wells):
        """
        Calculate antibody titer based on positive wells with logical validation

        使用初始抗体效价参数进行计算
        假设从左到右为系列稀释：1:10, 1:20, 1:40, 1:80, 1:160, 1:320, 1:640...

        判读规则：
        1. 首先判断最右侧孔（质控孔）必须为阴性
        2. 如果最左侧的第一个孔为阴性，那么这一行的结果均判读为阴性
        3. 如果出现逻辑错误（如：第一个孔阴性，但后面有阳性），则添加备注
        4. 如果第一个孔阳性，则找到最后一个连续阳性孔作为效价终点
        """
        if not wells:
            print("⚠️  No wells provided for titer calculation")
            return []

        # Create a matrix of predictions
        predictions_matrix = np.zeros((self.grid_rows, self.grid_cols))
        confidence_matrix = np.zeros((self.grid_rows, self.grid_cols))

        for well in wells:
            if 'prediction' not in well:
                print(f"⚠️  Well at position {well.get('position')} has no prediction")
                continue

            row, col = well['position']
            predictions_matrix[row, col] = 1 if well['prediction'] == 'POS' else 0
            confidence_matrix[row, col] = well.get('confidence', 0.5)

        # Calculate titer for each row (antibody type)
        titers = []
        for row in range(self.grid_rows):
            row_predictions = predictions_matrix[row, :]
            row_confidences = confidence_matrix[row, :]

            # 1. 检查最右侧孔（质控孔）是否为阴性
            control_well_col = self.grid_cols - 1  # 最右侧列为质控孔
            control_is_negative = row_predictions[control_well_col] == 0

            if not control_is_negative:
                # 质控孔阳性，整个行无效
                titers.append({
                    'row': row + 1,
                    'dilution': "Invalid",
                    'last_positive_column': 0,
                    'average_confidence': 0,
                    'interpretation': "Control well positive - Invalid result",
                    'notes': "Control well positive - Test result invalid, please re-test",
                    'is_valid': False
                })
                continue

            # 2. 检查最左侧第一个孔（最高浓度孔）
            first_well_col = 0  # 最左侧列
            first_well_is_negative = row_predictions[first_well_col] == 0

            # 3. 检查是否有逻辑错误：第一个孔阴性，但后面有阳性
            has_logic_error = False
            if first_well_is_negative:
                # 检查后面是否有阳性孔
                for col in range(1, self.grid_cols):
                    if row_predictions[col] == 1:
                        has_logic_error = True
                        break

            if first_well_is_negative:
                if has_logic_error:
                    # 逻辑错误：第一个孔阴性但后面有阳性
                    titers.append({
                        'row': row + 1,
                        'dilution': "Negative (Logic Error)",
                        'last_positive_column': 0,
                        'average_confidence': 0,
                        'interpretation': "Negative - Logical error detected",
                        'notes': "Test results illogical: First well negative but subsequent wells positive. Please re-shoot or verify testing procedure",
                        'is_valid': False
                    })
                else:
                    # 正常阴性结果：第一个孔阴性且后续全阴性
                    titers.append({
                        'row': row + 1,
                        'dilution': "Negative",
                        'last_positive_column': 0,
                        'average_confidence': 0,
                        'interpretation': "Negative - No antibody detected",
                        'notes': "Normal negative result",
                        'is_valid': True
                    })
                continue

            # 4. 第一个孔阳性，找到最后一个连续阳性孔
            last_positive_col = -1
            for col in range(self.grid_cols):
                if row_predictions[col] == 1:
                    last_positive_col = col
                else:
                    # 遇到第一个阴性孔就停止（系列稀释应为连续的）
                    break

            if last_positive_col >= 0:
                # 计算有效阳性孔的置信度平均值
                positive_confidences = row_confidences[:last_positive_col + 1]
                average_confidence = np.mean(positive_confidences) if len(positive_confidences) > 0 else 0

                # 使用初始抗体效价计算稀释倍数
                dilution_factor = self.initial_titer * (2 ** last_positive_col)

                # 检查是否有孤立的阳性孔（逻辑错误）
                has_isolated_positive = False
                if last_positive_col < self.grid_cols - 1:
                    for col in range(last_positive_col + 1, self.grid_cols):
                        if row_predictions[col] == 1:
                            has_isolated_positive = True
                            break

                if has_isolated_positive:
                    titers.append({
                        'row': row + 1,
                        'dilution': f"1:{dilution_factor} (Logic Error)",
                        'last_positive_column': last_positive_col + 1,
                        'average_confidence': average_confidence,
                        'interpretation': self.get_titer_interpretation(dilution_factor),
                        'notes': "Test results illogical: Positive wells not consecutive. Please re-shoot or verify testing procedure",
                        'is_valid': False
                    })
                else:
                    titers.append({
                        'row': row + 1,
                        'dilution': f"1:{dilution_factor}",
                        'last_positive_column': last_positive_col + 1,
                        'average_confidence': average_confidence,
                        'interpretation': self.get_titer_interpretation(dilution_factor),
                        'notes': "Normal positive result",
                        'is_valid': True
                    })
            else:
                # 理论上不会到达这里，因为已经检查过第一个孔是阳性
                titers.append({
                    'row': row + 1,
                    'dilution': "Error",
                    'last_positive_column': 0,
                    'average_confidence': 0,
                    'interpretation': "Analysis error",
                    'notes': "Error occurred during analysis",
                    'is_valid': False
                })

        return titers

    def get_titer_interpretation(self, dilution_factor: Union[int, str]) -> str:
        """
        根据稀释倍数给出抗体效价解释

        Args:
            dilution_factor: 稀释倍数或字符串

        Returns:
            解释字符串
        """
        # 如果是字符串（包含逻辑错误标记），直接返回
        if isinstance(dilution_factor, str):
            if "Logic Error" in dilution_factor or "Invalid" in dilution_factor:
                return "Test result needs review"

        try:
            # 尝试转换为整数
            if isinstance(dilution_factor, str):
                # 从字符串中提取数字，如 "1:160 (Logic Error)" -> 提取160
                import re
                numbers = re.findall(r'\d+', dilution_factor)
                if numbers:
                    factor = int(numbers[-1])
                else:
                    return "Invalid dilution factor"
            else:
                factor = int(dilution_factor)

            # 根据稀释倍数给出解释
            if factor < 40:
                return "Low antibody level"
            elif 40 <= factor < self.high_titer_threshold:
                return "Moderate antibody level"
            elif self.high_titer_threshold <= factor < self.high_titer_threshold * 4:  # 例如640
                return "High antibody level"
            else:
                return "Very high antibody level"

        except (ValueError, TypeError):
            return "Uninterpretable dilution factor"

    def get_image_quality_assessment(self, detection_result) -> Dict[str, str]:
        """
        根据检测结果获取图片质量评估

        Args:
            detection_result: 检测结果字典

        Returns:
            包含质量评估信息的字典
        """
        if detection_result is None or 'quality_check' not in detection_result:
            return {
                'quality_level': 'Unknown',
                'description': 'Image quality: Unknown',
                'suggestion': 'Unable to assess image quality',
                'confidence': 0.0
            }

        quality_check = detection_result['quality_check']
        if 'quality_rating' in quality_check:
            return quality_check['quality_rating']

        # 向后兼容：如果quality_rating不存在，根据置信度计算
        confidence = quality_check.get('metrics', {}).get('confidence', 0.0)
        return self.region_detector.assess_image_quality(confidence)

    def visualize_complete_analysis(self, image, reaction_region, wells, titers=None, save_path=None):
        """Visualize complete analysis results with larger fonts"""
        # 获取图片质量评估
        quality_assessment = self.get_image_quality_assessment(reaction_region)

        # Increase figure size for larger fonts
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))

        # Increase global font size
        plt.rcParams.update({'font.size': 14})

        # 1. Original image with reaction region detection
        axes[0, 0].imshow(image)
        if reaction_region:
            x, y, w, h = reaction_region['bbox']
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue', linewidth=3)
            axes[0, 0].add_patch(rect)

            # Show region confidence
            quality_check = reaction_region.get('quality_check', {})
            metrics = quality_check.get('metrics', {})
            confidence = metrics.get('confidence', reaction_region.get('confidence', 0))

            # 添加图片质量标签
            quality_color = 'red' if confidence < 0.7 else 'orange' if confidence < 0.9 else 'green'
            axes[0, 0].text(0.02, 0.98, f'Quality: {quality_assessment["quality_level"]}',
                            transform=axes[0, 0].transAxes, fontsize=14,
                            color='white', backgroundcolor=quality_color,
                            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                               facecolor=quality_color,
                                                               alpha=0.8))

            axes[0, 0].set_title(f'🔍 Reaction Region Detection (Confidence: {confidence:.3f})',
                                 fontsize=20, fontweight='bold', pad=25)
        axes[0, 0].axis('off')

        # 2. Reaction well grid division
        axes[0, 1].imshow(image)
        if reaction_region:
            x, y, w, h = reaction_region['bbox']
            # Draw grid lines
            well_height = h // self.grid_rows
            well_width = w // self.grid_cols

            # Horizontal lines
            for i in range(self.grid_rows + 1):
                y_line = y + i * well_height
                axes[0, 1].axhline(y=y_line, color='white', linewidth=2, alpha=0.8)
            # Vertical lines
            for j in range(self.grid_cols + 1):
                x_line = x + j * well_width
                axes[0, 1].axvline(x=x_line, color='white', linewidth=2, alpha=0.8)

            # Label rows and columns
            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    center_x = x + j * well_width + well_width // 2
                    center_y = y + i * well_height + well_height // 2
                    axes[0, 1].text(center_x, center_y, f'{i + 1},{j + 1}',
                                    fontsize=10, ha='center', va='center', color='yellow',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

            axes[0, 1].set_title(f'📊 {self.grid_rows}×{self.grid_cols} Reaction Well Grid',
                                 fontsize=20, fontweight='bold', pad=25)
        axes[0, 1].axis('off')

        # 3. Deep learning prediction results
        axes[1, 0].imshow(image)
        if wells:
            pos_count = sum(1 for well in wells if well.get('prediction') == 'POS')
            neg_count = sum(1 for well in wells if well.get('prediction') == 'NEG')

            for well in wells:
                x, y, w, h = well['global_bbox']
                color = well.get('color', 'gray')
                confidence = well.get('confidence', 0)
                prediction = well.get('prediction', 'UNK')

                rect = plt.Rectangle((x, y), w, h, fill=False,
                                     edgecolor=color, linewidth=2, alpha=0.8)
                axes[1, 0].add_patch(rect)

                # Show result in well center
                center_x, center_y = x + w // 2, y + h // 2
                text_color = 'red' if prediction == 'POS' else 'green'
                axes[1, 0].text(center_x, center_y, f'{prediction}\n{confidence:.2f}',
                                fontsize=8, ha='center', va='center', color=text_color,
                                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.9))

            axes[1, 0].set_title(f'🧬 Positive/Negative Prediction (Positive: {pos_count}, Negative: {neg_count})',
                                 fontsize=20, fontweight='bold', pad=25)
        axes[1, 0].axis('off')

        # 4. Statistical information and titer results
        axes[1, 1].axis('off')
        if wells:
            # Basic statistics
            pos_count = sum(1 for well in wells if well.get('prediction') == 'POS')
            neg_count = len(wells) - pos_count

            info_text = (
                f"📈 ANALYSIS STATISTICS:\n\n"
                f"• Grid layout: {self.grid_rows}×{self.grid_cols}\n"
                f"• Total wells: {len(wells)}\n"
                f"• Positive wells: {pos_count}\n"
                f"• Negative wells: {neg_count}\n"
                f"• Positive rate: {pos_count / len(wells) * 100:.1f}%\n"
                f"• Initial antibody titer: 1:{self.initial_titer}\n\n"
            )

            # 添加图片质量信息
            info_text += f"📷 IMAGE QUALITY ASSESSMENT:\n\n"
            info_text += f"• {quality_assessment['description']}\n"
            info_text += f"• Detection confidence: {quality_assessment['confidence']:.3f}\n"
            info_text += f"• Suggestion: {quality_assessment['suggestion']}\n\n"

            if self.predictor.model is not None:
                confidences = [well.get('confidence', 0) for well in wells]
                avg_confidence = np.mean(confidences)
                info_text += f"• Average prediction confidence: {avg_confidence:.3f}\n"
                info_text += f"• High confidence(>0.9): {sum(c > 0.9 for c in confidences)} wells\n"
                info_text += f"• Low confidence(<0.7): {sum(c < 0.7 for c in confidences)} wells\n\n"

            # Add titer results
            if titers:
                info_text += f"🧪 ANTIBODY TITER RESULTS:\n\n"
                for titer in titers:
                    info_text += f"• Row {titer['row']}: {titer['dilution']}"
                    if titer.get('interpretation'):
                        info_text += f" ({titer['interpretation']})"
                    if titer['average_confidence'] > 0:
                        info_text += f" [Confidence: {titer['average_confidence']:.3f}]"

                    # 添加备注信息
                    if titer.get('notes'):
                        info_text += f"\n   Note: {titer['notes']}"

                    # 标记无效结果
                    if not titer.get('is_valid', True):
                        info_text += f" ❌"

                    info_text += "\n"

            axes[1, 1].text(0.1, 0.95, info_text, transform=axes[1, 1].transAxes,
                            fontsize=16, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout(pad=5.0)

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
            print(f"💾 Analysis results saved: {save_path}")

        plt.show()

    def analyze_complete_image(self, image_path, visualize=True, save_dir=None):
        """Complete image analysis pipeline"""
        print(f"\n{'=' * 60}")
        print(f"🔍 Starting image analysis: {os.path.basename(image_path)}")
        print(f"{'=' * 60}")

        # 1. Image preprocessing
        image = self.preprocess_image(image_path)

        # 2. Reaction region detection (using Phase 1 method)
        print("📋 Step 1: Reaction region detection...")
        reaction_region, black_text_mask = self.region_detector.detect_reaction_region_complete(image)

        if not reaction_region:
            print("❌ No reaction region detected, analysis terminated")
            return None

        print(f"✅ Reaction region detected: {reaction_region['bbox']}")
        print(f"✅ Detection confidence: {reaction_region.get('confidence', 0):.3f}")

        # 3. 获取并显示图片质量评估
        quality_assessment = self.get_image_quality_assessment(reaction_region)
        print(f"📷 Image quality assessment: {quality_assessment['quality_level']}")
        print(f"   {quality_assessment['description']}")
        print(f"   {quality_assessment['suggestion']}")

        # 3. Create reaction wells based on grid parameters
        print("📋 Step 2: Creating reaction well grid...")
        wells = self.create_wells_from_grid(image, reaction_region)

        if not wells:
            print("❌ No reaction wells created, analysis terminated")
            return None

        # 4. Deep learning prediction
        print("📋 Step 3: Deep learning prediction...")
        wells = self.analyze_wells_with_dl(wells)

        # 5. Calculate antibody titer (使用初始抗体效价参数)
        print("📋 Step 4: Calculating antibody titer...")
        titers = self.calculate_antibody_titer(wells)

        # 6. Visualize results
        if visualize:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = os.path.join(save_dir, f"complete_analysis_{base_name}.png")
            else:
                save_path = None

            self.visualize_complete_analysis(image, reaction_region, wells, titers, save_path)

        # 7. Return complete results
        result = {
            'image_path': image_path,
            'grid_layout': f"{self.grid_rows}×{self.grid_cols}",
            'initial_titer': self.initial_titer,
            'reaction_region': reaction_region,
            'image_quality': quality_assessment,
            'wells_count': len(wells),
            'positive_count': sum(1 for well in wells if well.get('prediction') == 'POS'),
            'negative_count': sum(1 for well in wells if well.get('prediction') == 'NEG'),
            'wells': wells,
            'antibody_titers': titers
        }

        print(f"\n🎉 Analysis completed!")
        print(f"   Grid layout: {self.grid_rows}×{self.grid_cols}")
        print(f"   Initial titer: 1:{self.initial_titer}")
        print(f"   Image quality: {quality_assessment['quality_level']}")
        print(f"   Positive wells: {result['positive_count']}")
        print(f"   Negative wells: {result['negative_count']}")
        print(f"   Positive rate: {result['positive_count'] / result['wells_count'] * 100:.1f}%")

        print(f"\n🧪 Antibody Titer Results:")
        for titer in titers:
            print(f"   Row {titer['row']}: {titer['dilution']}")
            if titer.get('interpretation'):
                print(f"        Interpretation: {titer['interpretation']}")
            if titer.get('notes'):
                print(f"        Note: {titer['notes']}")
            if not titer.get('is_valid', True):
                print(f"        ⚠️  Warning: This result needs review!")

        print(f"\n📋 Detailed prediction results (first 10 wells):")
        for i, well in enumerate(result['wells'][:10]):
            row, col = well['position']
            pred = well['prediction']
            conf = well['confidence']
            print(f"   Position ({row + 1},{col + 1}): {pred} (Confidence: {conf:.3f})")

        if len(result['wells']) > 10:
            print(f"   ... and {len(result['wells']) - 10} more wells")

        return result


# Usage example
if __name__ == "__main__":
    # Initialize analyzer with grid layout parameters and initial titer
    analyzer = ELISA_Complete_Analyzer(
        grid_rows=4,
        grid_cols=6,
        initial_titer=40,
        high_titer_threshold=320,  # 现在可以自定义高抗体效价阈值，如1:320
        model_path="elisa_classifier_final.pth"
    )

    # Analyze single image
    image_path = "6.jpg"  # Replace with your image path

    if os.path.exists(image_path):
        result = analyzer.analyze_complete_image(
            image_path,
            visualize=True,
            save_dir="complete_results_english"
        )
    else:
        print(f"❌ Image file does not exist: {image_path}")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📁 Files in current directory: {os.listdir('.')[:10]}")