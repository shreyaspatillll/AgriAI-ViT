"""
AgriAI-ViT BERT-Based Recommendation Engine
Generates actionable agricultural recommendations using BERT
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from typing import Dict, List, Optional
import json
from pathlib import Path
import numpy as np


class BERTRecommendationEngine:
      """
          BERT-based recommendation engine for crop disease management
              Generates actionable recommendations based on detected diseases
                  """

    def __init__(self, model_name: str = 'bert-base-uncased', device: str = None):
              """
                      Initialize BERT Recommendation Engine

                                      Args:
                                                  model_name: Pretrained BERT model name
                                                              device: Device to run model on (cuda/cpu)
                                                                      """
              self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
              self.tokenizer = BertTokenizer.from_pretrained(model_name)
              self.model = BertModel.from_pretrained(model_name).to(self.device)
              self.model.eval()

        # Load recommendation database
              self.recommendations_db = self._load_recommendations_database()

    def _load_recommendations_database(self) -> Dict:
              """
                      Load curated recommendations for different crop diseases

                                      Returns:
                                                  recommendations_db: Dictionary mapping diseases to recommendations
                                                          """
              recommendations = {
                  # Rice Diseases
                  "Rice_Bacterial_Leaf_Blight": {
                      "disease_name": "Bacterial Leaf Blight",
                      "crop": "Rice",
                      "severity": "High",
                      "symptoms": "Water-soaked lesions on leaf tips and margins, turning yellow to white",
                      "treatment": [
                          "Apply copper-based bactericides (e.g., Copper hydroxide at 2g/L)",
                          "Use certified disease-free seeds",
                          "Spray Streptocycline (200 ppm) + Copper oxychloride (2.5 g/L)",
                          "Avoid excessive nitrogen fertilization"
                      ],
                      "prevention": [
                          "Plant resistant varieties like Improved Samba Mahsuri (ISM)",
                          "Maintain proper plant spacing (20cm x 15cm)",
                          "Ensure good field drainage",
                          "Remove and destroy infected plant debris",
                          "Practice crop rotation with non-host crops"
                      ],
                      "cultural_practices": [
                          "Avoid overhead irrigation during wet season",
                          "Apply balanced NPK fertilizers",
                          "Monitor field regularly for early detection"
                      ],
                      "estimated_yield_loss": "20-50% if untreated",
                      "management_cost": "₹2,000-3,000 per acre",
                      "recovery_time": "2-3 weeks with proper treatment"
                  },

                  "Rice_Brown_Spot": {
                      "disease_name": "Brown Spot",
                      "crop": "Rice",
                      "severity": "Medium",
                      "symptoms": "Small circular brown spots with gray centers on leaves",
                      "treatment": [
                          "Apply Mancozeb fungicide (2.5 g/L) at 10-day intervals",
                          "Use Carbendazim (1g/L) for severe infections",
                          "Spray Tricyclazole (0.6 g/L)",
                          "Apply potassium-rich fertilizers to boost plant immunity"
                      ],
                      "prevention": [
                          "Use disease-free certified seeds",
                          "Treat seeds with Carbendazim (2g/kg seed)",
                          "Maintain optimal soil nutrition (avoid potassium deficiency)",
                          "Ensure proper water management",
                          "Plant resistant varieties"
                      ],
                      "cultural_practices": [
                          "Avoid water stress during critical growth stages",
                          "Apply potassium fertilizers (Muriate of Potash 30 kg/acre)",
                          "Remove infected leaves and stubble after harvest"
                      ],
                      "estimated_yield_loss": "10-30% if untreated",
                      "management_cost": "₹1,500-2,500 per acre",
                      "recovery_time": "2-4 weeks"
                  },

                  "Rice_Leaf_Blast": {
                      "disease_name": "Leaf Blast",
                      "crop": "Rice",
                      "severity": "High",
                      "symptoms": "Diamond-shaped lesions with brown margins and gray centers",
                      "treatment": [
                          "Apply Tricyclazole (0.6 g/L) at first sign of disease",
                          "Use Carbendazim (1 g/L) alternating with Mancozeb",
                          "Spray Tebuconazole (1 mL/L) for severe cases",
                          "Apply silicon fertilizers to strengthen cell walls"
                      ],
                      "prevention": [
                          "Plant resistant varieties like Pusa Basmati 1509",
                          "Avoid excessive nitrogen application",
                          "Maintain proper plant spacing",
                          "Use seed treatment with Tricyclazole (2g/kg)",
                          "Destroy crop residues after harvest"
                      ],
                      "cultural_practices": [
                          "Split nitrogen application into 3-4 doses",
                          "Ensure good field drainage",
                          "Avoid late evening irrigation",
                          "Monitor weather and apply preventive sprays before rainy periods"
                      ],
                      "estimated_yield_loss": "30-60% if untreated",
                      "management_cost": "₹2,500-4,000 per acre",
                      "recovery_time": "3-5 weeks"
                  },

                  "Rice_Leaf_Scald": {
                      "disease_name": "Leaf Scald",
                      "crop": "Rice",
                      "severity": "Medium",
                      "symptoms": "Large scalded lesions with wavy borders on leaves",
                      "treatment": [
                          "Apply Propiconazole (1 mL/L) at disease onset",
                          "Use Mancozeb (2.5 g/L) as preventive spray",
                          "Spray Carbendazim + Mancozeb combination",
                          "Improve field drainage immediately"
                      ],
                      "prevention": [
                          "Plant resistant varieties",
                          "Ensure proper field leveling and drainage",
                          "Avoid water stagnation",
                          "Use balanced fertilization",
                          "Practice proper crop rotation"
                      ],
                      "cultural_practices": [
                          "Maintain optimal water levels (no stagnation)",
                          "Remove infected plant parts",
                          "Apply organic matter to improve soil health"
                      ],
                      "estimated_yield_loss": "15-35% if untreated",
                      "management_cost": "₹1,800-2,800 per acre",
                      "recovery_time": "2-3 weeks"
                  },

                  "Rice_Narrow_Brown_Spot": {
                      "disease_name": "Narrow Brown Spot",
                      "crop": "Rice",
                      "severity": "Low to Medium",
                      "symptoms": "Narrow brown lesions parallel to leaf veins",
                      "treatment": [
                          "Apply Mancozeb (2.5 g/L) preventively",
                          "Use Copper oxychloride (2.5 g/L) if needed",
                          "Ensure adequate potassium nutrition",
                          "Improve overall plant vigor"
                      ],
                      "prevention": [
                          "Maintain balanced soil fertility",
                          "Avoid potassium deficiency",
                          "Use disease-free seeds",
                          "Practice good water management",
                          "Plant tolerant varieties"
                      ],
                      "cultural_practices": [
                          "Apply potassium fertilizers (30 kg MOP/acre)",
                          "Ensure proper plant nutrition",
                          "Avoid water stress"
                      ],
                      "estimated_yield_loss": "5-15% if untreated",
                      "management_cost": "₹1,000-2,000 per acre",
                      "recovery_time": "1-2 weeks"
                  },

                  "Rice_Healthy": {
                      "disease_name": "Healthy Crop",
                      "crop": "Rice",
                      "severity": "None",
                      "symptoms": "No disease symptoms detected",
                      "treatment": [],
                      "prevention": [
                          "Continue regular monitoring",
                          "Maintain optimal nutrition and water management",
                          "Practice preventive sprays during disease-prone seasons",
                          "Keep field clean and weed-free"
                      ],
                      "cultural_practices": [
                          "Apply recommended doses of NPK fertilizers",
                          "Maintain proper water levels",
                          "Monitor for pest and disease regularly",
                          "Practice crop rotation"
                      ],
                      "estimated_yield_loss": "0%",
                      "management_cost": "Regular maintenance: ₹500-1,000 per acre",
                      "recovery_time": "N/A"
                  },

                  # Wheat Diseases (Extensible)
                  "Wheat_Leaf_Rust": {
                      "disease_name": "Leaf Rust",
                      "crop": "Wheat",
                      "severity": "High",
                      "symptoms": "Orange-red pustules on leaves",
                      "treatment": [
                          "Apply Propiconazole (0.1%) at first rust appearance",
                          "Use Tebuconazole (0.1%) for severe cases",
                          "Spray Mancozeb + Carbendazim combination",
                          "Ensure 2-3 sprays at 10-day intervals"
                      ],
                      "prevention": [
                          "Plant rust-resistant varieties like HD 3086, PBW 343",
                          "Avoid late sowing",
                          "Remove volunteer wheat plants",
                          "Use certified disease-free seeds",
                          "Practice crop rotation"
                      ],
                      "cultural_practices": [
                          "Maintain optimal sowing time",
                          "Ensure balanced fertilization",
                          "Destroy crop residues after harvest",
                          "Monitor weather for rust-favorable conditions"
                      ],
                      "estimated_yield_loss": "30-70% if untreated",
                      "management_cost": "₹2,000-3,500 per acre",
                      "recovery_time": "3-4 weeks"
                  },

                  "Wheat_Yellow_Rust": {
                      "disease_name": "Yellow Rust (Stripe Rust)",
                      "crop": "Wheat",
                      "severity": "Very High",
                      "symptoms": "Yellow pustules arranged in stripes on leaves",
                      "treatment": [
                          "Apply Propiconazole (0.1%) immediately",
                          "Use Tebuconazole (0.1%) at 10-day intervals",
                          "Spray Mancozeb (0.25%) as protectant",
                          "Apply systemic fungicides for rapid control"
                      ],
                      "prevention": [
                          "Plant resistant varieties like PBW 550, HD 2967",
                          "Follow timely sowing (avoid early sowing)",
                          "Remove alternate hosts and volunteer plants",
                          "Use hot water seed treatment (52°C for 10 min)",
                          "Monitor and apply preventive sprays"
                      ],
                      "cultural_practices": [
                          "Avoid excessive nitrogen fertilization",
                          "Ensure proper plant spacing",
                          "Remove infected plants early",
                          "Scout fields weekly during cool weather"
                      ],
                      "estimated_yield_loss": "40-80% if untreated",
                      "management_cost": "₹2,500-4,500 per acre",
                      "recovery_time": "4-6 weeks"
                  },

                  "Wheat_Healthy": {
                      "disease_name": "Healthy Crop",
                      "crop": "Wheat",
                      "severity": "None",
                      "symptoms": "No disease symptoms detected",
                      "treatment": [],
                      "prevention": [
                          "Continue regular field monitoring",
                          "Maintain optimal irrigation and nutrition",
                          "Practice preventive sprays if weather is conducive to disease",
                          "Keep field weed-free"
                      ],
                      "cultural_practices": [
                          "Apply recommended NPK doses",
                          "Ensure 4-5 irrigations at critical stages",
                          "Monitor for rust diseases during cool weather",
                          "Practice timely harvesting"
                      ],
                      "estimated_yield_loss": "0%",
                      "management_cost": "Regular maintenance: ₹800-1,500 per acre",
                      "recovery_time": "N/A"
                  }
              }

        return recommendations

    def generate_recommendation(self, disease_class: str, 
                                                               confidence: float = None,
                                                               region: str = "General",
                                                               season: str = "Kharif") -> Dict:
                                                                         """
                                                                                 Generate comprehensive recommendation for detected disease

                                                                                                 Args:
                                                                                                             disease_class: Detected disease class name
                                                                                                                         confidence: Model prediction confidence
                                                                                                                                     region: Geographic region (for localized recommendations)
                                                                                                                                                 season: Cropping season (Kharif/Rabi)
                                                                                                                                                             
                                                                                                                                                                     Returns:
                                                                                                                                                                                 recommendation: Comprehensive recommendation dictionary
                                                                                                                                                                                         """
                                                                         # Normalize disease class name
                                                                         disease_key = disease_class.replace(" ", "_").replace("-", "_")

        # Get base recommendation
        if disease_key in self.recommendations_db:
                      recommendation = self.recommendations_db[disease_key].copy()
else:
              # Fallback for unknown diseases
              recommendation = self._generate_generic_recommendation(disease_class)

        # Add metadata
          recommendation['detection_confidence'] = f"{confidence*100:.2f}%" if confidence else "N/A"
        recommendation['region'] = region
        recommendation['season'] = season
        recommendation['recommendation_id'] = f"{disease_key}_{region}_{season}"

        # Generate formatted text summary
        recommendation['summary'] = self._format_recommendation_summary(recommendation)

        return recommendation

    def _generate_generic_recommendation(self, disease_class: str) -> Dict:
              """
                      Generate generic recommendation for unknown disease classes

                                      Args:
                                                  disease_class: Disease class name

                                                                      Returns:
                                                                                  recommendation: Generic recommendation dictionary
                                                                                          """
              return {
                  "disease_name": disease_class,
                  "crop": "Unknown",
                  "severity": "Unknown",
                  "symptoms": "Please consult local agricultural extension officer",
                  "treatment": [
                      "Contact nearest Krishi Vigyan Kendra (KVK)",
                      "Send sample to plant pathology lab for confirmation",
                      "Isolate affected plants to prevent spread",
                      "Document symptoms with photographs"
                  ],
                  "prevention": [
                      "Maintain good field hygiene",
                      "Practice crop rotation",
                      "Use certified disease-free seeds",
                      "Monitor field regularly"
                  ],
                  "cultural_practices": [
                      "Ensure balanced nutrition",
                      "Maintain proper irrigation",
                      "Remove plant debris"
                  ],
                  "estimated_yield_loss": "Unknown - requires expert assessment",
                  "management_cost": "Variable",
                  "recovery_time": "Unknown"
              }

    def _format_recommendation_summary(self, recommendation: Dict) -> str:
              """
                      Format recommendation into readable text summary

                                      Args:
                                                  recommendation: Recommendation dictionary

                                                                      Returns:
"""
AgriAI-ViT BERT-Based Recommendation Engine
Generates actionable agricultural recommendations using BERT
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from typing import Dict, List, Optional
import json
from pathlib import Path
import numpy as np


class BERTRecommendationEngine:
    """
    BERT-based recommendation engine for crop disease management
    Generates actionable recommendations based on detected diseases
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = None):
        """
        Initialize BERT Recommendation Engine
        
        Args:
            model_name: Pretrained BERT model name
            device: Device to run model on (cuda/cpu)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Load recommendation database
        self.recommendations_db = self._load_recommendations_database()
    
    def _load_recommendations_database(self) -> Dict:
        """
        Load curated recommendations for different crop diseases
        
        Returns:
            recommendations_db: Dictionary mapping diseases to recommendations
        """
        recommendations = {
            # Rice Diseases
            "Rice_Bacterial_Leaf_Blight": {
                "disease_name": "Bacterial Leaf Blight",
                "crop": "Rice",
                "severity": "High",
                "symptoms": "Water-soaked lesions on leaf tips and margins, turning yellow to white",
                "treatment": [
                    "Apply copper-based bactericides (e.g., Copper hydroxide at 2g/L)",
                    "Use certified disease-free seeds",
                    "Spray Streptocycline (200 ppm) + Copper oxychloride (2.5 g/L)",
                    "Avoid excessive nitrogen fertilization"
                ],
                "prevention": [
                    "Plant resistant varieties like Improved Samba Mahsuri (ISM)",
                    "Maintain proper plant spacing (20cm x 15cm)",
                    "Ensure good field drainage",
                    "Remove and destroy infected plant debris",
                    "Practice crop rotation with non-host crops"
                ],
                "cultural_practices": [
                    "Avoid overhead irrigation during wet season",
                    "Apply balanced NPK fertilizers",
                    "Monitor field regularly for early detection"
                ],
                "estimated_yield_loss": "20-50% if untreated",
                "management_cost": "₹2,000-3,000 per acre",
                "recovery_time": "2-3 weeks with proper treatment"
            },
            
            "Rice_Brown_Spot": {
                "disease_name": "Brown Spot",
                "crop": "Rice",
                "severity": "Medium",
                "symptoms": "Small circular brown spots with gray centers on leaves",
                "treatment": [
                    "Apply Mancozeb fungicide (2.5 g/L) at 10-day intervals",
                    "Use Carbendazim (1g/L) for severe infections",
                    "Spray Tricyclazole (0.6 g/L)",
                    "Apply potassium-rich fertilizers to boost plant immunity"
                ],
                "prevention": [
                    "Use disease-free certified seeds",
                    "Treat seeds with Carbendazim (2g/kg seed)",
                    "Maintain optimal soil nutrition (avoid potassium deficiency)",
                    "Ensure proper water management",
                    "Plant resistant varieties"
                ],
                "cultural_practices": [
                    "Avoid water stress during critical growth stages",
                    "Apply potassium fertilizers (Muriate of Potash 30 kg/acre)",
                    "Remove infected leaves and stubble after harvest"
                ],
                "estimated_yield_loss": "10-30% if untreated",
                "management_cost": "₹1,500-2,500 per acre",
                "recovery_time": "2-4 weeks"
            },
            
            "Rice_Leaf_Blast": {
                "disease_name": "Leaf Blast",
                "crop": "Rice",
                "severity": "High",
                "symptoms": "Diamond-shaped lesions with brown margins and gray centers",
                "treatment": [
                    "Apply Tricyclazole (0.6 g/L) at first sign of disease",
                    "Use Carbendazim (1 g/L) alternating with Mancozeb",
                    "Spray Tebuconazole (1 mL/L) for severe cases",
                    "Apply silicon fertilizers to strengthen cell walls"
                ],
                "prevention": [
                    "Plant resistant varieties like Pusa Basmati 1509",
                    "Avoid excessive nitrogen application",
                    "Maintain proper plant spacing",
                    "Use seed treatment with Tricyclazole (2g/kg)",
                    "Destroy crop residues after harvest"
                ],
                "cultural_practices": [
                    "Split nitrogen application into 3-4 doses",
                    "Ensure good field drainage",
                    "Avoid late evening irrigation",
                    "Monitor weather and apply preventive sprays before rainy periods"
                ],
                "estimated_yield_loss": "30-60% if untreated",
                "management_cost": "₹2,500-4,000 per acre",
                "recovery_time": "3-5 weeks"
            },
            
            "Rice_Leaf_Scald": {
                "disease_name": "Leaf Scald",
                "crop": "Rice",
                "severity": "Medium",
                "symptoms": "Large scalded lesions with wavy borders on leaves",
                "treatment": [
                    "Apply Propiconazole (1 mL/L) at disease onset",
                    "Use Mancozeb (2.5 g/L) as preventive spray",
                    "Spray Carbendazim + Mancozeb combination",
                    "Improve field drainage immediately"
                ],
                "prevention": [
                    "Plant resistant varieties",
                    "Ensure proper field leveling and drainage",
                    "Avoid water stagnation",
                    "Use balanced fertilization",
                    "Practice proper crop rotation"
                ],
                "cultural_practices": [
                    "Maintain optimal water levels (no stagnation)",
                    "Remove infected plant parts",
                    "Apply organic matter to improve soil health"
                ],
                "estimated_yield_loss": "15-35% if untreated",
                "management_cost": "₹1,800-2,800 per acre",
                "recovery_time": "2-3 weeks"
            },
            
            "Rice_Narrow_Brown_Spot": {
                "disease_name": "Narrow Brown Spot",
                "crop": "Rice",
                "severity": "Low to Medium",
                "symptoms": "Narrow brown lesions parallel to leaf veins",
                "treatment": [
                    "Apply Mancozeb (2.5 g/L) preventively",
                    "Use Copper oxychloride (2.5 g/L) if needed",
                    "Ensure adequate potassium nutrition",
                    "Improve overall plant vigor"
                ],
                "prevention": [
                    "Maintain balanced soil fertility",
                    "Avoid potassium deficiency",
                    "Use disease-free seeds",
                    "Practice good water management",
                    "Plant tolerant varieties"
                ],
                "cultural_practices": [
                    "Apply potassium fertilizers (30 kg MOP/acre)",
                    "Ensure proper plant nutrition",
                    "Avoid water stress"
                ],
                "estimated_yield_loss": "5-15% if untreated",
                "management_cost": "₹1,000-2,000 per acre",
                "recovery_time": "1-2 weeks"
            },
            
            "Rice_Healthy": {
                "disease_name": "Healthy Crop",
                "crop": "Rice",
                "severity": "None",
                "symptoms": "No disease symptoms detected",
                "treatment": [],
                "prevention": [
                    "Continue regular monitoring",
                    "Maintain optimal nutrition and water management",
                    "Practice preventive sprays during disease-prone seasons",
                    "Keep field clean and weed-free"
                ],
                "cultural_practices": [
                    "Apply recommended doses of NPK fertilizers",
                    "Maintain proper water levels",
                    "Monitor for pest and disease regularly",
                    "Practice crop rotation"
                ],
                "estimated_yield_loss": "0%",
                "management_cost": "Regular maintenance: ₹500-1,000 per acre",
                "recovery_time": "N/A"
            },
            
            # Wheat Diseases (Extensible)
            "Wheat_Leaf_Rust": {
                "disease_name": "Leaf Rust",
                "crop": "Wheat",
                "severity": "High",
                "symptoms": "Orange-red pustules on leaves",
                "treatment": [
                    "Apply Propiconazole (0.1%) at first rust appearance",
                    "Use Tebuconazole (0.1%) for severe cases",
                    "Spray Mancozeb + Carbendazim combination",
                    "Ensure 2-3 sprays at 10-day intervals"
                ],
                "prevention": [
                    "Plant rust-resistant varieties like HD 3086, PBW 343",
                    "Avoid late sowing",
                    "Remove volunteer wheat plants",
                    "Use certified disease-free seeds",
                    "Practice crop rotation"
                ],
                "cultural_practices": [
                    "Maintain optimal sowing time",
                    "Ensure balanced fertilization",
                    "Destroy crop residues after harvest",
                    "Monitor weather for rust-favorable conditions"
                ],
                "estimated_yield_loss": "30-70% if untreated",
                "management_cost": "₹2,000-3,500 per acre",
                "recovery_time": "3-4 weeks"
            },
            
            "Wheat_Yellow_Rust": {
                "disease_name": "Yellow Rust (Stripe Rust)",
                "crop": "Wheat",
                "severity": "Very High",
                "symptoms": "Yellow pustules arranged in stripes on leaves",
                "treatment": [
                    "Apply Propiconazole (0.1%) immediately",
                    "Use Tebuconazole (0.1%) at 10-day intervals",
                    "Spray Mancozeb (0.25%) as protectant",
                    "Apply systemic fungicides for rapid control"
                ],
                "prevention": [
                    "Plant resistant varieties like PBW 550, HD 2967",
                    "Follow timely sowing (avoid early sowing)",
                    "Remove alternate hosts and volunteer plants",
                    "Use hot water seed treatment (52°C for 10 min)",
                    "Monitor and apply preventive sprays"
                ],
                "cultural_practices": [
                    "Avoid excessive nitrogen fertilization",
                    "Ensure proper plant spacing",
                    "Remove infected plants early",
                    "Scout fields weekly during cool weather"
                ],
                "estimated_yield_loss": "40-80% if untreated",
                "management_cost": "₹2,500-4,500 per acre",
                "recovery_time": "4-6 weeks"
            },
            
            "Wheat_Healthy": {
                "disease_name": "Healthy Crop",
                "crop": "Wheat",
                "severity": "None",
                "symptoms": "No disease symptoms detected",
                "treatment": [],
                "prevention": [
                    "Continue regular field monitoring",
                    "Maintain optimal irrigation and nutrition",
                    "Practice preventive sprays if weather is conducive to disease",
                    "Keep field weed-free"
                ],
                "cultural_practices": [
                    "Apply recommended NPK doses",
                    "Ensure 4-5 irrigations at critical stages",
                    "Monitor for rust diseases during cool weather",
                    "Practice timely harvesting"
                ],
                "estimated_yield_loss": "0%",
                "management_cost": "Regular maintenance: ₹800-1,500 per acre",
                "recovery_time": "N/A"
            }
        }
        
        return recommendations
    
    def generate_recommendation(self, disease_class: str, 
                               confidence: float = None,
                               region: str = "General",
                               season: str = "Kharif") -> Dict:
        """
        Generate comprehensive recommendation for detected disease
        
        Args:
            disease_class: Detected disease class name
            confidence: Model prediction confidence
            region: Geographic region (for localized recommendations)
            season: Cropping season (Kharif/Rabi)
            
        Returns:
            recommendation: Comprehensive recommendation dictionary
        """
        # Normalize disease class name
        disease_key = disease_class.replace(" ", "_").replace("-", "_")
        
        # Get base recommendation
        if disease_key in self.recommendations_db:
            recommendation = self.recommendations_db[disease_key].copy()
        else:
            # Fallback for unknown diseases
            recommendation = self._generate_generic_recommendation(disease_class)
        
        # Add metadata
        recommendation['detection_confidence'] = f"{confidence*100:.2f}%" if confidence else "N/A"
        recommendation['region'] = region
        recommendation['season'] = season
        recommendation['recommendation_id'] = f"{disease_key}_{region}_{season}"
        
        # Generate formatted text summary
        recommendation['summary'] = self._format_recommendation_summary(recommendation)
        
        return recommendation
    
    def _generate_generic_recommendation(self, disease_class: str) -> Dict:
        """
        Generate generic recommendation for unknown disease classes
        
        Args:
            disease_class: Disease class name
            
        Returns:
            recommendation: Generic recommendation dictionary
        """
        return {
            "disease_name": disease_class,
            "crop": "Unknown",
            "severity": "Unknown",
            "symptoms": "Please consult local agricultural extension officer",
            "treatment": [
                "Contact nearest Krishi Vigyan Kendra (KVK)",
                "Send sample to plant pathology lab for confirmation",
                "Isolate affected plants to prevent spread",
                "Document symptoms with photographs"
            ],
            "prevention": [
                "Maintain good field hygiene",
                "Practice crop rotation",
                "Use certified disease-free seeds",
                "Monitor field regularly"
            ],
            "cultural_practices": [
                "Ensure balanced nutrition",
                "Maintain proper irrigation",
                "Remove plant debris"
            ],
            "estimated_yield_loss": "Unknown - requires expert assessment",
            "management_cost": "Variable",
            "recovery_time": "Unknown"
        }
    
    def _format_recommendation_summary(self, recommendation: Dict) -> str:
        """
        Format recommendation into readable text summary
        
        Args:
            recommendation: Recommendation dictionary
            
        Returns:
            summary: Formatted text summary
        """
        summary_parts = []
        
        # Header
        summary_parts.append(f"=== {recommendation['disease_name']} - {recommendation['crop']} ===\n")
        
        # Detection info
        if recommendation.get('detection_confidence'):
            summary_parts.append(f"Detection Confidence: {recommendation['detection_confidence']}")
        summary_parts.append(f"Severity: {recommendation['severity']}")
        summary_parts.append(f"Region: {recommendation['region']} | Season: {recommendation['season']}\n")
        
        # Symptoms
        summary_parts.append(f"SYMPTOMS:\n{recommendation['symptoms']}\n")
        
        # Treatment
        if recommendation['treatment']:
            summary_parts.append("IMMEDIATE TREATMENT:")
            for i, treatment in enumerate(recommendation['treatment'], 1):
                summary_parts.append(f"  {i}. {treatment}")
            summary_parts.append("")
        
        # Prevention
        if recommendation['prevention']:
            summary_parts.append("PREVENTION MEASURES:")
            for i, prevention in enumerate(recommendation['prevention'], 1):
                summary_parts.append(f"  {i}. {prevention}")
            summary_parts.append("")
        
        # Cultural practices
        if recommendation['cultural_practices']:
            summary_parts.append("CULTURAL PRACTICES:")
            for i, practice in enumerate(recommendation['cultural_practices'], 1):
                summary_parts.append(f"  {i}. {practice}")
            summary_parts.append("")
        
        # Economic impact
        summary_parts.append("ECONOMIC IMPACT:")
        summary_parts.append(f"  - Estimated Yield Loss: {recommendation['estimated_yield_loss']}")
        summary_parts.append(f"  - Management Cost: {recommendation['management_cost']}")
        summary_parts.append(f"  - Recovery Time: {recommendation['recovery_time']}\n")
        
        # Contact info
        summary_parts.append("For expert advice, contact:")
        summary_parts.append("  - Nearest Krishi Vigyan Kendra (KVK)")
        summary_parts.append("  - Agriculture Department Helpline: 1800-180-1551")
        summary_parts.append("  - Kisan Call Centre: 1800-180-1551")
        
        return "\n".join(summary_parts)
    
    def batch_generate_recommendations(self, 
                                      predictions: List[Dict]) -> List[Dict]:
        """
        Generate recommendations for batch predictions
        
        Args:
            predictions: List of prediction dictionaries with 'class' and 'confidence'
            
        Returns:
            recommendations: List of recommendation dictionaries
        """
        recommendations = []
        
        for pred in predictions:
            disease_class = pred.get('class', 'Unknown')
            confidence = pred.get('confidence', None)
            region = pred.get('region', 'General')
            season = pred.get('season', 'Kharif')
            
            recommendation = self.generate_recommendation(
                disease_class, confidence, region, season
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def save_recommendation(self, recommendation: Dict, save_path: str):
        """
        Save recommendation to JSON file
        
        Args:
            recommendation: Recommendation dictionary
            save_path: Path to save JSON file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(recommendation, f, indent=4, ensure_ascii=False)
        
        print(f"Recommendation saved to {save_path}")
    
    def get_emergency_contact_info(self, region: str = "General") -> Dict:
        """
        Get emergency contact information for agricultural support
        
        Args:
            region: Geographic region
            
        Returns:
            contact_info: Dictionary with contact information
        """
        contacts = {
            "National": {
                "Kisan Call Centre": "1800-180-1551",
                "Agriculture Department": "1800-180-1551",
                "PM-KISAN Helpline": "155261 / 011-24300606",
                "Crop Insurance": "1800-180-1551"
            },
            "Emergency": {
                "Pest/Disease Outbreak": "Contact nearest KVK immediately",
                "Weather Alert": "IMD Helpline: 1800-180-1551",
                "Market Information": "e-NAM Helpline: 1800-270-0224"
            },
            "Online Resources": {
                "KVK Portal": "https://kvk.icar.gov.in/",
                "AgriMarket": "https://agmarknet.gov.in/",
                "Crop Insurance": "https://pmfby.gov.in/",
                "Kisan Portal": "https://farmer.gov.in/"
            }
        }
        
        return contacts


# Example usage and testing
if __name__ == '__main__':
    print("AgriAI-ViT BERT Recommendation Engine")
    print("=" * 50)
    
    # Initialize engine
    engine = BERTRecommendationEngine()
    
    # Test recommendation generation
    test_disease = "Rice_Bacterial_Leaf_Blight"
    recommendation = engine.generate_recommendation(
        disease_class=test_disease,
        confidence=0.95,
        region="Punjab",
        season="Kharif"
    )
    
    print("\nGenerated Recommendation:")
    print(recommendation['summary'])
    
    # Save recommendation
    engine.save_recommendation(
        recommendation, 
        'recommendations/test_recommendation.json'
    )
    
    print("\n" + "=" * 50)
    print("Recommendation engine initialized successfully!")
                                                                                          """
              summary_parts = []

        # Header
              summary_parts.append(f"=== {recommendation['disease_name']} - {recommendation['crop']} ===\n")

        # Detection info
        if recommendation.get('detection_confidence'):
                      summary_parts.append(f"Detection Confidence: {recommendation['detection_confidence']}")
                  summary_parts.append(f"Severity: {recommendation['severity']}")
        summary_parts.append(f"Region: {recommendation['region']} | Season: {recommendation['season']}\n")

        # Symptoms
        summary_parts.append(f"SYMPTOMS:\n{recommendation['symptoms']}\n")

        # Treatment
        if recommendation['treatment']:
                      summary_parts.append("IMMEDIATE TREATMENT:")
                      for i, treatment in enumerate(recommendation['treatment'], 1):
                                        summary_parts.append(f"  {i}. {treatment}")
                                    summary_parts.append("")

        # Prevention
        if recommendation['prevention']:
                      summary_parts.append("PREVENTION MEASURES:")
            for i, prevention in enumerate(recommendation['prevention'], 1):
                              summary_parts.append(f"  {i}. {prevention}")
                          summary_parts.append("")

        # Cultural practices
        if recommendation['cultural_practices']:
                      summary_parts.append("CULTURAL PRACTICES:")
            for i, practice in enumerate(recommendation['cultural_practices'], 1):
                              summary_parts.append(f"  {i}. {practice}")
                          summary_parts.append("")

        # Economic impact
        summary_parts.append("ECONOMIC IMPACT:")
        summary_parts.append(f"  - Estimated Yield Loss: {recommendation['estimated_yield_loss']}")
        summary_parts.append(f"  - Management Cost: {recommendation['management_cost']}")
        summary_parts.append(f"  - Recovery Time: {recommendation['recovery_time']}\n")

        # Contact info
        summary_parts.append("For expert advice, contact:")
        summary_parts.append("  - Nearest Krishi Vigyan Kendra (KVK)")
        summary_parts.append("  - Agriculture Department Helpline: 1800-180-1551")
        summary_parts.append("  - Kisan Call Centre: 1800-180-1551")

        return "\n".join(summary_parts)

    def batch_generate_recommendations(self, 
                                                                             predictions: List[Dict]) -> List[Dict]:
                                                                                       """
                                                                                               Generate recommendations for batch predictions
                                                                                                       
                                                                                                               Args:
                                                                                                                           predictions: List of prediction dictionaries with 'class' and 'confidence'
                                                                                                                                       
                                                                                                                                               Returns:
                                                                                                                                                           recommendations: List of recommendation dictionaries
                                                                                                                                                                   """
                                                                                       recommendations = []

        for pred in predictions:
                      disease_class = pred.get('class', 'Unknown')
            confidence = pred.get('confidence', None)
            region = pred.get('region', 'General')
            season = pred.get('season', 'Kharif')

            recommendation = self.generate_recommendation(
                              disease_class, confidence, region, season
            )
            recommendations.append(recommendation)

        return recommendations

    def save_recommendation(self, recommendation: Dict, save_path: str):
              """
                      Save recommendation to JSON file

                                      Args:
                                                  recommendation: Recommendation dictionary
                                                              save_path: Path to save JSON file
                                                                      """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
                      json.dump(recommendation, f, indent=4, ensure_ascii=False)

        print(f"Recommendation saved to {save_path}")

    def get_emergency_contact_info(self, region: str = "General") -> Dict:
              """
                      Get emergency contact information for agricultural support

                                      Args:
                                                  region: Geographic region

                                                                      Returns:
                                                                                  contact_info: Dictionary with contact information
                                                                                          """
        contacts = {
                      "National": {
                                        "Kisan Call Centre": "1800-180-1551",
                                        "Agriculture Department": "1800-180-1551",
                                        "PM-KISAN Helpline": "155261 / 011-24300606",
                                        "Crop Insurance": "1800-180-1551"
                      },
                      "Emergency": {
                                        "Pest/Disease Outbreak": "Contact nearest KVK immediately",
                                        "Weather Alert": "IMD Helpline: 1800-180-1551",
                                        "Market Information": "e-NAM Helpline: 1800-270-0224"
                      },
                      "Online Resources": {
                                        "KVK Portal": "https://kvk.icar.gov.in/",
                                        "AgriMarket": "https://agmarknet.gov.in/",
                                        "Crop Insurance": "https://pmfby.gov.in/",
                                        "Kisan Portal": "https://farmer.gov.in/"
                      }
        }

        return contacts


#
