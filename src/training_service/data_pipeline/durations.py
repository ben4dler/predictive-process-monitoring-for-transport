import training_service.data_pipeline.activities as ac

BASE_DURATIONS = {
    ac.Activities.Abholung: (30,  90),
    ac.Activities.Depot_Eingang: (60,  180),
    ac.Activities.Zollabfertigung: (120, 480), 
    ac.Activities.Hauptlauf: (180, 720),
    ac.Activities.Auslieferung: (30,  90)
}