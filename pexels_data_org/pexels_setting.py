new_queries = [
    # Culinary and Food (Further Expanded)
    'Farmers Market Food Stalls', 'Vegan Recipes', 'Exotic Fruits and Vegetables', 'Cheese Platters', 
    'Barbecue Grilling', 'Luxury Wine and Dine', 'Pastry Chef Baking',

    # Adventure and Extreme Sports (Further Expanded)
    'Rock Climbing Expeditions', 'Extreme Skiing', 'Free Diving in Caves', 'Hang Gliding', 
    'Desert Dune Bashing', 'Whitewater Rafting', 'Cliff Diving',

    # Urban Life and Cityscapes (Further Expanded)
    'Neon Signs at Night', 'Vintage City Cars', 'Graffiti Art Walls', 'Street Performers in Cities', 
    'Busy Train Stations', 'Public Parks in Spring', 'Bicycle Commuters',

    # Crafts and DIY Projects (Further Expanded)
    'Woodworking Workshops', 'Pottery Making', 'Knitting Projects', 'Quilting Patterns', 
    'Beaded Jewelry Crafting', 'Handmade Soap', 'Watercolor Art',

    # Fantasy and Sci-Fi Themes (Further Expanded)
    'Alien Landscapes', 'Spaceships in Orbit', 'Dragons in Battle', 'Underwater Cities', 
    'Steampunk Machines', 'Galactic Wars', 'Enchanting Forests',

    # Seasons and Weather (Further Expanded)
    'Snowfall in Forests', 'Tropical Storm Clouds', 'Golden Sunrises', 'Springtime Flowers', 
    'Windy Days on Beaches', 'Summer Rain Showers', 'Frost on Leaves',

    # Historical and Cultural Heritage (Further Expanded)
    'Medieval Castles', 'Renaissance Art Museums', 'Victorian Architecture', 'Old Train Stations', 
    'Cultural Street Parades', 'Historical Mansions', 'Ancient Temples',

    # Fitness and Wellness (Further Expanded)
    'Meditation in Nature', 'Stand-Up Paddleboarding', 'Yoga on the Beach', 'Cycling in Countryside', 
    'CrossFit Competitions', 'Outdoor Hiking Trails', 'Spa and Wellness Centers',

    # Technology and Innovation (Further Expanded)
    'Autonomous Cars', 'Drone Deliveries', 'Augmented Reality Apps', 'Solar Energy Farms', 
    'Wind Turbines in Action', '3D Printing', 'Smart City Infrastructure'
]



settings_file = 'pexels_settings.txt'

# Write the new queries to settings.txt
with open(settings_file, 'w') as file:
    for query in new_queries:
        file.write(f"{query}\n")

print(f"Queries have been saved to {settings_file}")