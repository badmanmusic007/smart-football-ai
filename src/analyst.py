def get_key_factors(features, prediction, home_team, away_team):
    """
    Analyzes features and prediction probabilities to generate human-readable insights.
    """
    insights = []
    
    # ELO Difference
    elo_gap = features[9] - features[10]
    if abs(elo_gap) > 150:
        winner = home_team if elo_gap > 0 else away_team
        insights.append(f"Significant ELO Mismatch: {winner} has a rating advantage of {abs(int(elo_gap))} points.")

    # Form Difference
    form_gap = features[0] - features[1]
    if abs(form_gap) > 6: # More than 2 wins difference in form
        winner = home_team if form_gap > 0 else away_team
        insights.append(f"Form Disparity: {winner} is in much better recent form.")

    # Fortress vs. Traveler
    home_venue_form = features[2]
    away_venue_form = features[3]
    if home_venue_form >= 10 and away_venue_form <= 3:
        insights.append("Classic Fortress vs. Traveler: Strong home team meets a struggling away side.")

    # Goal Machine vs. Leaky Defense
    home_scored = features[5]
    away_conceded = features[8]
    if home_scored > 2.2 and away_conceded > 1.8:
        insights.append(f"Goal Alert: {home_team}'s potent attack faces {away_team}'s leaky defense.")

    # Referee Factor
    ref_harshness = features[21]
    if ref_harshness > 4.5:
        insights.append("Card Magnet Match: The appointed referee is significantly stricter than average, expect bookings.")
        
    # High-Confidence Prediction
    if prediction['home_win_prob'] > 0.75 or prediction['away_win_prob'] > 0.75:
        winner = home_team if prediction['home_win_prob'] > 0.75 else away_team
        insights.append(f"High Confidence Pick: The model is unusually confident in a {winner} victory.")
        
    if not insights:
        insights.append("Tight Contest: No single factor strongly dominates the prediction. Expect a close match.")

    return insights[:3] # Return top 3 insights
