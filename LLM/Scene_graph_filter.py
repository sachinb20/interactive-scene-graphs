import json

# Load JSON data from file
with open('scene_graph.json', 'r') as f:
    data = json.load(f)

# Extract desired keys and values
result = {}
for key, value in data.items():
    result[key] = {
        "parentReceptacles": value["parentReceptacles"],
        "ObjectState": value["ObjectState"]
    }

# Save result to a new JSON filez
with open('filtered_scene_graph.json', 'w') as f:
    json.dump(result, f, indent=2)

print("Filtered JSON data saved to filtered_scene_graph.json")




{
  "Agent": {
    "state": "clear",
    "contains": []
  },
  "Fridge|-02.48|+00.00|-00.78": {

    "state": "Closed",
    "contains": ["Egg|-02.53|+00.60|-00.71"]
    
  },
  "Sink|+01.38|+00.81|-01.27": {
    "state": "clear",

    "contains": [
      "Mug|+01.45|+00.91|-01.23" , "Tomato|+01.30|+00.96|-01.08" , "Lettuce|+01.11|+00.83|-01.43" ,"DishSponge|+01.74|+00.90|-00.86"
    ]
  },

  "Microwave|-02.58|+00.90|+02.44": {

    "state": "Closed",
    "contains": []
  },

  "CounterTop|-01.49|+00.95|+01.32":{

    "state": "clear",
    "contains": [ "Plate|-02.35|+00.90|+00.05","Potato|-02.24|+00.94|-00.18","Spatula|-02.31|+00.91|+00.33"]
  },

  "Cabinet|-02.15|+00.40|+00.70":{

    "state": "Closed",
    "contains": ["Pot|-02.31|+00.11|+00.89"]
  },

"CounterTop|+00.47|+00.95|-01.63":
{

    "state": "clear",
    "contains": ["Kettle|+00.85|+00.90|-01.79","SoapBottle|+01.02|+00.90|-01.65","PaperTowelRoll|+00.69|+01.01|-01.83"]
},

"CounterTop|+01.59|+00.95|+00.41":{

  "state": "clear",
  "contains": ["Fork|+01.44|+00.90|+00.34","SaltShaker|+01.67|+00.90|+00.45","ButterKnife|+01.44|+00.90|+00.43","PepperShaker|+01.76|+00.90|+00.37"]
},

"CounterTop|-00.36|+00.95|+01.09":{

      "state": "clear",
      "contains": ["Pan|+00.00|+00.90|+00.95","Knife|-00.64|+00.91|+01.62","Apple|-00.48|+00.97|+00.41","Bowl|-00.65|+00.90|+01.26","Bread|-00.71|+00.98|+00.43", "Cup|-00.65|+00.90|+00.74","Spoon|-00.66|+00.96|+01.33"]

},
"Drawer|-02.28|+00.79|+01.37":
{

      "state": "Closed",
      "contains": []
},

"Cabinet|+00.15|+02.01|-01.60":
{

      "state": "Closed",
      "contains": []
}
}
