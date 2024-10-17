new_queries = [
    # City and Urban Life
    'Skylines at Sunset', 'Urban Graffiti', 'City Life at Dawn', 'Busy Crosswalks', 
    'City Traffic', 'Street Performers', 'Urban Green Spaces', 'City Rooftops', 
    'Downtown Streets', 'Skyscrapers at Night',

    # Animals and Wildlife
    'Savannah Wildlife', 'Ocean Life', 'Polar Animals', 'Endangered Species', 
    'Rainforest Creatures', 'Birds of Prey', 'Desert Wildlife', 'Pet Animals', 
    'Nocturnal Animals', 'Mountain Animals',

    # People and Portraits
    'Diverse Portraits', 'Festival Crowds', 'Urban Explorers', 'People in Nature', 
    'Portraits in Black and White', 'Smiling Faces', 'Fashionable Individuals', 'Outdoor Fitness Enthusiasts', 
    'Business Professionals', 'Artistic Portraits',

    # Travel and Adventure
    'Exotic Islands', 'City Breaks', 'Hiking Mountains', 'Tropical Beaches', 
    'Desert Safari', 'Road Trips', 'Backpacking Adventures', 'Waterfall Treks', 
    'Winter Sports', 'Historical Landmarks',

    # Food and Drink
    'Gourmet Breakfasts', 'Farm-to-Table Dishes', 'Artisan Cocktails', 'Coffee Art', 
    'Seafood Platters', 'Street Snacks', 'Pastry Creations', 'Colorful Smoothies', 
    'Food Truck Specials', 'Ice Cream Desserts',

    # Technology and Gadgets
    'Augmented Reality Devices', 'Tech Startups', 'Smart Home Gadgets', 'Virtual Meetings', 
    'Drone Photography', 'Tech Innovators', 'Robotics in Action', 'Smartphone Photography', 
    'Electric Vehicles', 'Cutting-Edge Technology',

    # Fashion and Clothing
    'Street Fashion', 'Beachwear', 'Outdoor Fashion Shoots', 'Elegant Evening Wear', 
    'Accessories in Focus', 'Layered Winter Styles', 'Spring Dresses', 'Formal Suits', 
    'Leather Boots', 'Bold Fashion Choices',

    # Sports and Fitness
    'Competitive Sports', 'Outdoor Yoga', 'Cycling Events', 'Beach Running', 
    'Skateboarding', 'Marathons', 'Rock Climbing', 'Football Games', 
    'Swimming Competitions', 'Gym Sessions',

    # Vehicles and Transportation
    'High-End Sports Cars', 'Eco-Friendly Vehicles', 'Motorbikes on the Road', 'Subway Systems', 
    'Private Jets', 'Public Transport Systems', 'Ferry Boats', 'Electric Bikes', 
    'Hot Air Balloons', 'Luxury RVs'
]
# File path for settings.txt
settings_file = 'pexels_settings.txt'
with open(settings_file, 'w') as file:
    for query in new_queries:
        file.write(f"{query}\n")

print(f"Queries have been saved to {settings_file}")