from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import pickle
from pathlib import Path
from src.config import settings

class MatchPredictor:
    def __init__(self):
        # Optimized Hyperparameters from Tuning (Phase 18)
        params = {
            'n_estimators': 200,
            'learning_rate': 0.01,
            'max_depth': 5,
            'subsample': 0.9,
            'min_samples_split': 2,
            'random_state': settings.RANDOM_SEED
        }

        # Model 1: Match Result (Home/Draw/Away)
        self.model_res = GradientBoostingClassifier(**params)
        
        # Goal Models (1.5, 2.5, 3.5, 4.5)
        self.model_ou15 = GradientBoostingClassifier(**params)
        self.model_ou25 = GradientBoostingClassifier(**params)
        self.model_ou35 = GradientBoostingClassifier(**params)
        self.model_ou45 = GradientBoostingClassifier(**params)
        
        # BTTS Model (Both Teams To Score)
        self.model_btts = GradientBoostingClassifier(**params)
        
        # Corners Model (Over/Under 9.5)
        self.model_corners = GradientBoostingClassifier(**params)
        
        # Cards Model (Over/Under 3.5)
        self.model_cards = GradientBoostingClassifier(**params)

        self.is_trained = False
        self.load_model()

    def train(self, X_train, y_res, y_ou15, y_ou25, y_ou35, y_ou45, y_btts, y_corners, y_cards):
        """
        Train all models.
        """
        print("Training Smart Football AI Models...")
        self.model_res.fit(X_train, y_res)
        
        print("Training Goal Models...")
        self.model_ou15.fit(X_train, y_ou15)
        self.model_ou25.fit(X_train, y_ou25)
        self.model_ou35.fit(X_train, y_ou35)
        self.model_ou45.fit(X_train, y_ou45)
        
        print("Training BTTS Model...")
        self.model_btts.fit(X_train, y_btts)
        
        print("Training Corners Model...")
        self.model_corners.fit(X_train, y_corners)
        
        print("Training Cards Model...")
        self.model_cards.fit(X_train, y_cards)
        
        self.is_trained = True
        self.save_model()

    def predict(self, features):
        """
        Predict outcome probabilities for all markets.
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
            
        df = pd.DataFrame([features], columns=settings.FEATURES)
        
        probs_res = self.model_res.predict_proba(df)[0]
        
        # Get probabilities for Over (index 1)
        prob_over_15 = self.model_ou15.predict_proba(df)[0][1]
        prob_over_25 = self.model_ou25.predict_proba(df)[0][1]
        prob_over_35 = self.model_ou35.predict_proba(df)[0][1]
        prob_over_45 = self.model_ou45.predict_proba(df)[0][1]
        prob_btts_yes = self.model_btts.predict_proba(df)[0][1]
        prob_corners_over = self.model_corners.predict_proba(df)[0][1]
        prob_cards_over = self.model_cards.predict_proba(df)[0][1]
        
        return {
            "home_win_prob": float(probs_res[0]),
            "draw_prob": float(probs_res[1]),
            "away_win_prob": float(probs_res[2]),
            
            "over_15_prob": float(prob_over_15),
            "over_25_prob": float(prob_over_25),
            "over_35_prob": float(prob_over_35),
            "over_45_prob": float(prob_over_45),
            "btts_prob": float(prob_btts_yes),
            "corners_over_prob": float(prob_corners_over),
            "cards_over_prob": float(prob_cards_over)
        }

    def save_model(self):
        pickle.dump(self.model_res, open(settings.MODELS_DIR / "model_res.pkl", "wb"))
        pickle.dump(self.model_ou15, open(settings.MODELS_DIR / "model_ou15.pkl", "wb"))
        pickle.dump(self.model_ou25, open(settings.MODELS_DIR / "model_ou25.pkl", "wb"))
        pickle.dump(self.model_ou35, open(settings.MODELS_DIR / "model_ou35.pkl", "wb"))
        pickle.dump(self.model_ou45, open(settings.MODELS_DIR / "model_ou45.pkl", "wb"))
        pickle.dump(self.model_btts, open(settings.MODELS_DIR / "model_btts.pkl", "wb"))
        pickle.dump(self.model_corners, open(settings.MODELS_DIR / "model_corners.pkl", "wb"))
        pickle.dump(self.model_cards, open(settings.MODELS_DIR / "model_cards.pkl", "wb"))

    def load_model(self):
        try:
            self.model_res = pickle.load(open(settings.MODELS_DIR / "model_res.pkl", "rb"))
            self.model_ou15 = pickle.load(open(settings.MODELS_DIR / "model_ou15.pkl", "rb"))
            self.model_ou25 = pickle.load(open(settings.MODELS_DIR / "model_ou25.pkl", "rb"))
            self.model_ou35 = pickle.load(open(settings.MODELS_DIR / "model_ou35.pkl", "rb"))
            self.model_ou45 = pickle.load(open(settings.MODELS_DIR / "model_ou45.pkl", "rb"))
            self.model_btts = pickle.load(open(settings.MODELS_DIR / "model_btts.pkl", "rb"))
            self.model_corners = pickle.load(open(settings.MODELS_DIR / "model_corners.pkl", "rb"))
            self.model_cards = pickle.load(open(settings.MODELS_DIR / "model_cards.pkl", "rb"))
            self.is_trained = True
        except FileNotFoundError:
            self.is_trained = False