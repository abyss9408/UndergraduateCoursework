package edu.singaporetech.oipmemo

object data {
    val momentsList = listOf(
        MemorableData(
            R.drawable.redmond,
            "DigiPen@Redmond Best Moments",
            "\uD83D\uDCF8 A generation Z, with phone in hand, captures the spectacle for the digital scrolls—a snapshot of the heavens in their evening attire. \"Sunset vibes at SIT, y'all! \uD83C\uDF07✨ #GoldenHourMagic #SITSunsetDreams\\",
        ),
        MemorableData(
            R.drawable.bilbao,
            "DigiPen@Bilbao Best Moments",
            "As the sun sets over DigiPen@Bilbao, the sky transforms into a vivid canvas of emojis, painting a tapestry of hues that even the coolest Instagram filter can't quite capture. \uD83C\uDF07✨ Gen Z students gather, phones in hand, snapping pics of the sun's descent, turning the moment into a social media masterpiece. Hashtagging the sky's color palette, they celebrate the end of the day with a burst of emojis and a touch of Gen Z flair. \uD83C\uDF06\uD83D\uDCF8 #SunsetVibes #SITSPSunsetMagic",
        ),
        // Add more MemorableData Samples, don't delete the above two
        MemorableData(
            R.drawable.bilbao,
            "Sweet Tooth Chronicles",
            "🍦 Two friends, Jenny and Alex, embark on a quest for the perfect scoop of ice cream. The sun is blazing, and the heat is on! They decide to hit the coolest ice cream spot in town, 'Chill Haven.' With excitement in their eyes, they debate the ultimate question: cone or cup? 🤔🍨 #IceCreamAdventure #ChillHavenEscapade",
        ),
        MemorableData(
            R.drawable.redmond,
            "Caffeine Chronicles",
            "☕️ Sara and Mark, self-proclaimed coffee aficionados, embark on a quest for the perfect espresso. Armed with passion and a love for latte art, they explore hidden coffee gems in the city, sipping and savoring each velvety cup. #CoffeeAdventure #EspressoExplorers",
        ),
        MemorableData(
            R.drawable.bilbao,
            "Serene Escape",
            "🌿 Emily and James retreat to a hidden cabin in the woods for a weekend of tranquility. Surrounded by nature's symphony, they trade city chaos for the soothing melody of chirping birds and rustling leaves. #NatureGetaway #CabinRetreat",
        ),
        MemorableData(
            R.drawable.redmond,
            "Worldly Rhythms",
            "🌍 Lucy and Carlos embark on a global music journey, discovering the beats of different cultures. From Brazilian samba to Indian classical, they create a playlist that transcends borders. #GlobalGroove #MusicalOdyssey",
        ),
        // make more MemorableData Samples
        MemorableData(
            R.drawable.singapore,
            "Urban Garden Oasis",
            "🌺 Mia discovers a hidden rooftop garden in the bustling city, where time slows down and worries fade. Surrounded by vibrant blooms and buzzing bees, she finds peace in the urban jungle. #RooftopRetreat #CityGarden",
        ),
        MemorableData(
            R.drawable.redmond,
            "Stargazing Adventures",
            "⭐ Ben and Luna drive far from city lights to witness the cosmos in its full glory. With telescopes ready and hot cocoa in hand, they trace constellations and make wishes on shooting stars. #StarryNight #CosmicWonders",
        ),
        MemorableData(
            R.drawable.bilbao,
            "Bookworm Paradise",
            "📚 Oliver spends hours in the cozy corner of an indie bookstore, lost in tales of far-off lands and epic adventures. The smell of old pages and fresh coffee creates the perfect reading atmosphere. #BookLovers #ReadingNook",
        ),
        MemorableData(
            R.drawable.singapore,
            "Street Food Safari",
            "🍜 Maya and Ryan embark on a culinary journey through night markets, tasting exotic flavors and discovering local delicacies. Every bite tells a story of tradition and culture. #FoodieAdventure #NightMarket",
        ),
        MemorableData(
            R.drawable.redmond,
            "Hiking Heights",
            "🏔️ The trail leads higher as Sophie conquers her fears, reaching the summit where clouds meet earth. The view from the top makes every step worth it. #MountainVibes #SummitSuccess",
        ),
        MemorableData(
            R.drawable.bilbao,
            "Jazz Night Chronicles",
            "🎷 The smooth sounds of saxophones and trumpets fill the dimly lit jazz club as Kate and Tom sway to the rhythm. Live music has never felt so alive. #JazzLounge #LiveMusic",
        )
    )

    val campusList = listOf(
        MemorableData(
            R.drawable.redmond,
            "Redmond",
            "",
        ),
        MemorableData(
            R.drawable.bilbao,
            "Bilbao",
            "",
        ),
        MemorableData(
            R.drawable.singapore,
            "Singapore",
            ""
        )
    )
}

data class MemorableData(
    val imgDrawable: Int = R.drawable.redmond,
    val title: String = "DigiPen@Redmond Best Moments",
    val description: String = "\uD83D\uDCF8 A generation Z, with phone in hand, captures the spectacle for the digital scrolls—a snapshot of the heavens in their evening attire. \"Sunset vibes at SIT, y'all! \uD83C\uDF07✨ #GoldenHourMagic #SITSunsetDreams\\",
)