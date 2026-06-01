import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import training_service.data_pipeline.data_generator as dg
import training_service.data_pipeline.activities as activities
import training_service.data_pipeline.durations as durations
import training_service.data_pipeline.saving_service as saving_service
from utils.schemas import EventSchema
import pydantic

def simulate_case(case_id, start_time):
    events = []
    current_time = start_time
    
    activities_list = [a.name for a in activities.Activities]
    if random.random() > 0.6:
        activities_list.remove("Zollabfertigung")
    
    # Ein Wert von 0.8 bedeutet 20% schnellere Abwicklung, 1.5 bedeutet 50% langsamer.
    driver_reliability = random.uniform(0.8, 1.3) 
    resource = f"Fahrer_{random.randint(1, 20)}"

    accumulated_delay = 0.0
    total_steps = len(activities_list)

    for i, activity in enumerate(activities_list):
        weekday = current_time.weekday()
        hour    = current_time.hour
        weather = dg.generate_weather()
        traffic = dg.generate_traffic_index(weekday, hour)
        
        # Basis-Dauer
        activity_enum = getattr(activities.Activities, activity)
        base_duration = random.randint(*durations.BASE_DURATIONS[activity_enum])
        
        # Wenn wir schon 30 Min zu spät sind, erhöht das die Chance auf weitere 10% Delay
        chain_reaction = 1.1 if accumulated_delay > 30 else 1.0
        
        # Delay berechnen mit Fahrer-Faktor und Kettenreaktion
        delay = dg.calculate_delay(activity, weather, traffic, weekday)
        total_duration = (base_duration + delay) * driver_reliability * chain_reaction
        
        # Effektive Verspätung für dieses Segment
        actual_delay = total_duration - base_duration
        accumulated_delay += actual_delay
        black_swan_event, black_swan_delay = dg.generate_black_swan()

        raw_event = {
            "activity":          activity,
            "step_index":        i + 1,          
            "percent_done":      round((i+1)/total_steps, 2),
            "timestamp":         current_time,
            "weather_condition": weather,
            "traffic_index":     traffic,
            "weekday":           weekday,
            "is_holiday":        int(dg.is_holiday(current_time)),
            "delay_so_far":      round(accumulated_delay, 1),
            "driver_factor":     round(driver_reliability, 2),
            "black_swan_event":  black_swan_event if black_swan_event else "none",
            "black_swan_delay":  round(black_swan_delay, 1),
        }

        validated_event = EventSchema(**raw_event).model_dump()

        validated_event["case_id"] = case_id
        validated_event["resource"] = resource

        current_time += timedelta(minutes=total_duration)

        events.append(validated_event)
    
    return events

def generate_event_log(num_cases=10000, start_date="2024-01-01"):
    all_events = []
    base_date  = datetime.strptime(start_date, "%Y-%m-%d")
    
    for i in range(num_cases):
        case_id = f"CASE_{i+1:04d}"
        
        # Zufälliger Startzeitpunkt innerhalb eines Jahres
        offset     = timedelta(
            days    = random.randint(0, 365),
            hours   = random.randint(6, 18),  # nur Tageszeiten
            minutes = random.randint(0, 59)
        )
        start_time = base_date + offset
        
        events = simulate_case(case_id, start_time)
        all_events.extend(events)
    
    df = pd.DataFrame(all_events)
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    return df

# Generieren und speichern
log = generate_event_log(num_cases=10000)
saving_service.save_event_log(log, "transport_event_log.csv")
print(f"Generiert: {len(log)} Events aus {log['case_id'].nunique()} Cases")

def add_training_labels(df):
    """Fügt die Zielvariablen für alle 4 Vorhersageaufgaben hinzu"""
    
    result = []
    
    for case_id, case in df.groupby("case_id"):
        case      = case.reset_index(drop=True)
        last_time = case["timestamp"].iloc[-1]
        
        for i, row in case.iterrows():
            # 1. Remaining Time (Regression)
            remaining_time = (last_time - row["timestamp"]).total_seconds() / 60
            
            # 2. Next Activity (Klassifikation)
            if i+1 < len(case):
                next_activity = (case.iloc[i+1]["activity"])
            else:
                next_activity = ("END")

            # 3. Outcome: pünktlich? (Klassifikation)
            final_delay   = case.iloc[-1]["delay_so_far"]
            if final_delay <= 30:       # 30 Min Puffer
                on_time = 1 
            else: 
                on_time = 0  
            
            # 4. Risikowert: Wahrsch. für Verspätung > 60 Min (Regression 0-1)
            risk_score    = min(final_delay / 300, 1.0)
            
            result.append({
                **row.to_dict(),
                "label_remaining_time": round(remaining_time, 1),
                "label_next_activity":  next_activity,
                "label_on_time":        on_time,
                "label_risk_score":     round(risk_score, 2)
            })
    
    return pd.DataFrame(result)

# Labels hinzufügen
log_with_labels = add_training_labels(log)
saving_service.save_event_log(log_with_labels, "transport_event_log_labeled.csv")