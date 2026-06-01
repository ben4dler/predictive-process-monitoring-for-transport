import random
import numpy as np

def generate_weather():
    """Gewichtete Wetterverteilung je nach Jahreszeit"""
    conditions = ["klar", "regen", "schnee", "nebel"]
    weights    = [0.60,   0.25,    0.10,    0.05]
    return random.choices(conditions, weights=weights)[0]

def generate_traffic_index(weekday, hour):
    """Realistischer Stauindex basierend auf Wochentag & Uhrzeit"""
    base = 0.3
    
    # Montag und Freitag mehr Stau
    if weekday in [0, 4]:
        base += 0.2
    
    # Rushhour
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        base += 0.3
    
    # Wochenende weniger Stau
    if weekday >= 5:
        if 8 <= hour <= 10 or 15 <= hour <= 18:
            base += 0.2
        base -= 0.2
    
    # Zufällige Schwankung
    noise = np.random.normal(0, 0.05)
    return round(min(max(base + noise, 0.0), 1.0), 2)

AT_HOLIDAYS = [
    "2024-01-01", "2024-04-01", "2024-05-01",
    "2024-10-26", "2024-12-25", "2024-12-26"
]

def is_holiday(date):
    return date.strftime("%Y-%m-%d") in AT_HOLIDAYS

def calculate_delay(activity, weather, traffic_index, weekday):
    """
    Berechnet zusätzliche Verzögerung basierend auf Kontextfaktoren.
    Gibt Verzögerung in Minuten zurück.
    """
    delay = 0.0
    
    # Wettereinfluss
    weather_delays = {
        "klar":   0,
        "regen":  random.uniform(10, 30),
        "schnee": random.uniform(30, 120),
        "nebel":  random.uniform(15, 45)
    }
    delay += weather_delays.get(weather, 0)
    
    # Verkehrseinfluss
    delay += traffic_index * random.uniform(20, 60)
    
    # Zoll ist besonders unvorhersehbar
    if activity == "Zollabfertigung":
        if random.random() < 0.3:  # 30% Chance auf extra Verzögerung
            delay += random.uniform(60, 240)
    
    # Montagseffekt
    if weekday == 0:
        delay *= 1.2
    
    return round(delay, 1)

BLACK_SWAN_EVENTS = [
    ("Unfall",        0.02, 120, 360),  # 2% Chance, +2–6h
    ("Grenzsperre",   0.01, 240, 720),  # 1% Chance, +4–12h
    ("Fahrzeugpanne", 0.03,  60, 180),  # 3% Chance, +1–3h
    ("Streik",        0.005,480,1440),  # 0.5% Chance, +8–24h
]

def generate_black_swan():
    for name, prob, min_d, max_d in BLACK_SWAN_EVENTS:
        if random.random() < prob:
            return name, random.uniform(min_d, max_d)
    return None, 0.0