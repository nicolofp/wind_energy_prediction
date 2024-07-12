Wind power prediction
================

## Introduction

Dataset description for each of the four turbines:

- **Time**: hour of the day when readings occurred
- **temperature_2m**: temperature in degrees Fahrenheit at 2 meters
  above the surface
- **relativehumidity_2m**: relative humidity (as a percentage) at 2
  meters above the surface
- **dewpoint_2m**: dew point in degrees Fahrenheit at 2 meters above the
  surface
- **windspeed_10m**: wind speed in meters per second at 10 meters above
  the surface
- **windspeed_100m**: wind speed in meters per second at 100 meters
  above the surface
- **winddirection_10m**: wind direction in degrees (0-360) at 10 meters
  above the surface
- **winddirection_100m**: wind direction in degrees (0-360) at 100
  meters above the surface
- **windgusts_10m**: wind gusts in meters per second at 100 meters above
  the surface
- **Power**: turbine output, normalized to be between 0 and 1 (i.e., a
  percentage of maximum potential output)

``` python
import pandas as pd
```
