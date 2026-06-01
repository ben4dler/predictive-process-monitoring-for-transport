from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class EventSchema(BaseModel):
    activity:          str
    step_index:        int
    percent_done:      float
    timestamp:         Optional[datetime] = None
    weather_condition: str
    traffic_index:     float
    weekday:           int
    is_holiday:        int
    delay_so_far:      float
    driver_factor:     float
    black_swan_event:  str
    black_swan_delay:  float

class PredictionRequest(BaseModel):
    case_id:  str
    resource: str
    prefix:   List[EventSchema]