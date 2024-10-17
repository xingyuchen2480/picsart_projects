new_queries = [
    # Nature and Landscapes (Further Expanded)
    'Volcanic Eruptions', 'Foggy Forests', 'Desert Oasis', 'Mountain Rivers', 
    'Icy Glaciers', 'Canyons at Sunset', 'Lush Rainforests', 'Winding River Valleys',
    'Tidal Waves', 'Golden Wheat Fields',

    # City and Urban Life (Further Expanded)
    'Rooftop Gardens', 'Street Performances', 'Skyscrapers in Fog', 'Industrial Cityscapes', 
    'Riverside Walkways', 'Crowded Bazaars', 'Quiet Suburban Streets', 'Nightlife in City Centers',
    'Street Food Vendors', 'Pedestrian Crossings',

    # Animals and Wildlife (Further Expanded)
    'Pandas Eating Bamboo', 'Kangaroos in the Outback', 'Horses Galloping', 'Penguins in Antarctica',
    'Dolphins Jumping', 'Peacocks Displaying Feathers', 'Bald Eagles Soaring', 'Wolves in Snow',
    'Otters Playing in Rivers', 'Bears Catching Fish',

    # People and Portraits (Further Expanded)
    'Cultural Dance Performances', 'Students Graduating', 'Pilgrims on Journeys', 'Athletes Preparing for Races', 
    'Street Artists Painting', 'Families Enjoying Picnics', 'People on Road Trips', 'Street Food Chefs Cooking',
    'Artists Sketching Outdoors', 'Sunbathers on Beaches',

    # Travel and Adventure (Further Expanded)
    'Exploring Ancient Ruins', 'Skydiving Over Oceans', 'Jungle Treks', 'Climbing Frozen Waterfalls',
    'Exploring Remote Villages', 'Road Trips Across Deserts', 'Island Hopping Adventures', 'Exploring Ancient Castles',
    'Cross-Country Cycling', 'Trekking Through Jungle Trails',

    # Food and Drink (Further Expanded)
    'Artisan Bakeries', 'Sushi Platters', 'Coffee Art', 'Exotic Fruit Markets', 
    'Cheese Tastings', 'Barbecue Cookouts', 'Breakfast Buffets', 'Street Food Trucks',
    'Wine Tastings in Vineyards', 'Baking Desserts',

    # Technology and Gadgets (Further Expanded)
    '3D Printers in Action', 'Electric Cars Charging', 'Augmented Reality Glasses', 'Next-Gen Laptops',
    'Autonomous Delivery Robots', 'Wearable Fitness Trackers', 'Smart Light Bulbs', 'Foldable Smartphones',
    'Self-Driving Cars', 'Digital Art Tablets',

    # Fashion and Clothing (Further Expanded)
    'Runway Fashion Shows', 'Vintage Sunglasses', 'Denim Jackets', 'Ethnic Festival Attire', 
    'Winter Scarves and Hats', 'Summer Dresses', 'High Heels on Display', 'Luxury Watches',
    'Handmade Jewelry', 'Eco-Friendly Fashion',

    # Sports and Fitness (Further Expanded)
    'Mountain Biking Down Trails', 'Martial Arts Tournaments', 'Sailing Competitions', 'Rowing Teams',
    'Ice Hockey Matches', 'Ballet Performances', 'Cyclists in Races', 'Yoga on Clifftops',
    'Archery Competitions', 'Long-Distance Running',

    # Vehicles and Transportation (Further Expanded)
    'Bullet Trains in Motion', 'Cargo Ships Docking', 'Hot Air Balloons Over Countryside', 'Submarines Underwater',
    'Electric Cars on Highways', 'Vintage Motorcycles', 'Airplanes Taking Off', 'Luxury SUVs',
    'City Bicycles', 'Sailboats in the Ocean'
]


# File path for settings.txt
settings_file = 'unsplash_settings.txt'

# Write the new queries to settings.txt
with open(settings_file, 'w') as file:
    for query in new_queries:
        file.write(f"{query}\n")

print(f"Queries have been saved to {settings_file}")