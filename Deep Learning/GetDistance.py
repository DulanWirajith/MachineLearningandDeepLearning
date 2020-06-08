# from math import sin, cos, sqrt, atan2, radians
#
# # approximate radius of earth in km
# R = 6373.0
#
# lat1 = radians(80.42512)
# lon1 = radians(8.35546)
# lat2 = radians(80.42548)
# lon2 = radians(8.35529)
#
# dlon = lon2 - lon1
# dlat = lat2 - lat1
#
# a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
# c = 2 * atan2(sqrt(a), sqrt(1 - a))
#
# distance = R * c
#
# print("Result:", distance)
# print("Should be:", 278.546, "km")


import geopy.distance

coords_1 = (8.35546,80.42512)
coords_2 = (8.35529,80.42548)

print(geopy.distance.vincenty(coords_1, coords_2))