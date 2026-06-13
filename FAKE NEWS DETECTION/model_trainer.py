# model_trainer.py
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline

# 1. Sample production-grade training data (Real vs Fake patterns)
training_data = [
    ("Scientists discover new cure for diabetes using gene editing.", "REAL"),
    ("The government is hiding aliens in secret underground basements.", "FAKE"),
    ("Stock market hits record high amid economic growth reports.", "REAL"),
    ("Drinking lemon juice completely cures COVID-19 overnight.", "FAKE"),
    ("NASA launches new telescope to study distant exoplanets.", "REAL"),
    ("Celebrity spotted flying on a dragon over New York City.", "FAKE")
    ("Scientists discover new cure for diabetes using gene editing.", "REAL"),
    ("Stock market hits record high amid economic growth reports.", "REAL"),
    ("NASA launches new telescope to study distant exoplanets.", "REAL"),
    ("The Federal Reserve announced an unexpected interest rate cut today.", "REAL"),
    ("A major tech company unveiled its latest quantum computing processor.", "REAL"),
    ("Global climate summit reaches historic agreement on carbon emissions.", "REAL"),
    ("Archaeologists discover an intact ancient tomb in southern Egypt.", "REAL"),
    ("The World Health Organization clears a new vaccine for global distribution.", "REAL"),
    ("Local authorities open a new high-speed rail line connecting major cities.", "REAL"),
    ("A significant earthquake struck off the coast of Alaska early this morning.", "REAL"),
    ("Retail sales data shows a steady increase in consumer spending.", "REAL"),
    ("Deep-sea explorers discover three previously unknown marine species.", "REAL"),
    ("The national space agency successfully landed its rover on an asteroid.", "REAL"),
    ("A historic library completed the digitization of its entire manuscript collection.", "REAL"),
    ("Government officials sign a bilateral trade agreement to lower tariffs.", "REAL"),
    ("The city council approves funding for a massive urban forestry initiative.", "REAL"),
    ("Researchers develop a highly efficient solar cell using perovskite materials.", "REAL"),
    ("A premier medical journal published a breakthrough study on Alzheimer treatment.", "REAL"),
    ("Automobile manufacturers report a surge in electric vehicle production capacity.", "REAL"),
    ("International food security program delivers vital aid to drought-affected regions.", "REAL"),
    ("The government is hiding aliens in secret underground basements.", "FAKE"),
    ("Drinking lemon juice completely cures COVID-19 overnight.", "FAKE"),
    ("Celebrity spotted flying on a dragon over New York City.", "FAKE"),
    ("A secret satellite is broadcasting signals that control the weather globally.", "FAKE"),
    ("Eating chocolate every day makes humans completely immune to aging.", "FAKE"),
    ("Scientists confirm the moon is hollow and built by ancient civilizations.", "FAKE"),
    ("A smartphone app can now teleport physical objects across the room.", "FAKE"),
    ("The ocean will completely freeze over by the end of next week.", "FAKE"),
    ("Local zebra breaks out of zoo and wins an international chess tournament.", "FAKE"),
    ("Drinking water from gold cups instantly reverses baldness.", "FAKE"),
    ("A newly discovered plant species grows pure silver coins on its leaves.", "FAKE"),
    ("Major tech firm announces a software update that lets phones run without a battery.", "FAKE"),
    ("The Eiffel Tower was secretly moved to London overnight during a rainstorm.", "FAKE"),
    ("An invisible island has just been discovered in the middle of the Atlantic.", "FAKE"),
    ("Eating dynamic pizza crust allows individuals to speak fluent Italian instantly.", "FAKE"),
    ("A local cat was officially sworn in as the governor of a major state.", "FAKE"),
    ("Volcano in Antarctica erupts and shoots out millions of shiny diamonds.", "FAKE"),
    ("Gravity will be temporarily turned off for three minutes next Tuesday.", "FAKE"),
    ("A secret formula hidden in old paintings lets people walk through solid walls.", "FAKE"),
    ("Using an old coin as a phone screen protector grants unlimited cellular data.", "FAKE")
    ("Did a major world leader announce the immediate mandatory draft of all citizens?", "FAKE"),
    ("Is the central bank planning to completely ban cash transactions by next month?", "FAKE"),
    ("Will social media platforms begin charging a monthly fee to all users starting tomorrow?", "FAKE"),
    ("Did a viral video show a prominent politician using a body double during a recent rally?", "FAKE"),
    ("Can drinking a specific herbal solution cure chronic illnesses in less than 24 hours?", "FAKE"),
    ("Did a massive government agency confirm that a solar flare will permanently shut down the internet?", "FAKE"),
    ("Is it true that a new law allows authorities to monitor private phone cameras without a warrant?", "FAKE"),
    ("Did a major tech company accidentally release an artificial intelligence that gained sentience?", "FAKE"),
    ("Was a hidden camera system discovered inside the voting booths of a major election?", "FAKE"),
    ("Did a famous celebrity secretly pass away and get replaced by a lookalike actor?", "FAKE"),
    ("Is it true that emergency food rations are being stockpiled due to an incoming asteroid?", "FAKE"),
    ("Did a foreign military base launch a cyberattack that caused the recent national power outage?", "FAKE"),
    ("Are major supermarkets secretly mixing synthetic chemicals into regular table salt?", "FAKE"),
    ("Did a famous billionaire announce they are giving away half their fortune to random internet users?", "FAKE"),
    ("Is it true that a new global tax is being implemented on all digital bank transfers?", "FAKE"),
    ("Did a medical breakthrough confirm that standard tap water contains tracking microchips?", "FAKE"),
    ("Was an entire coastal town secretly evacuated due to an undisclosed chemical spill?", "FAKE"),
    ("Did a leaked document reveal that a major conflict was entirely staged by news networks?", "FAKE"),
    ("Are commercial flights spraying specific chemicals to alter weather patterns over urban areas?", "FAKE"),
    ("Did a top intelligence official confirm the existence of an underwater alien civilization?", "FAKE"),
    ("Did the world health agency declare a new international public health emergency?", "REAL"),
    ("Did the Supreme Court issue a landmark ruling regarding digital privacy rights?", "REAL"),
    ("Will the national space agency launch a manned mission to study the lunar surface?", "REAL"),
    ("Did a major electronics manufacturer recall thousands of laptops due to battery fire hazards?", "REAL"),
    ("Is the government implementing a new interest rate policy to curb rising inflation?", "REAL"),
    ("Did a powerful earthquake strike a major metropolitan area causing structural damage?", "REAL"),
    ("Are international leaders meeting at a global summit to discuss new trade tariffs?", "REAL"),
    ("Did researchers successfully use genetic therapy to treat a rare hereditary disorder?", "REAL"),
    ("Is a prominent technology firm facing an antitrust lawsuit filed by regulatory bodies?", "REAL"),
    ("Did a severe weather system trigger flash flood warnings across multiple coastal states?", "REAL"),
    ("Did the labor department report a significant shift in the national unemployment rate?", "REAL"),
    ("Is a major automotive brand transitioning its entire factory lineup to electric vehicles?", "REAL"),
    ("Did archeologists uncover a substantial collection of bronze age artifacts during a road project?", "REAL"),
    ("Are local authorities enforcing new zoning laws to increase affordable housing units?", "REAL"),
    ("Did a prestigious university publish a study linking high sugar consumption to heart disease?", "REAL"),
    ("Is the national transport authority upgrading the signaling system of the major rail network?", "REAL"),
    ("Did a major oil pipeline temporarily halt operations following a minor mechanical leak?", "REAL"),
    ("Are environmental agencies reporting a record increase in regional solar energy production?", "REAL"),
    ("Did a standard security update cause a temporary disruption in global airline check-in systems?", "REAL"),
    ("Is the city council allocating a new budget to upgrade municipal water filtration plants?", "REAL")
]

X, y = zip(*training_data)

# 2. Build an ML Pipeline (TF-IDF Vectorizer + Passive Aggressive Classifier)
# Passive Aggressive algorithms are excellent for text classification and massive datasets
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('model', PassiveAggressiveClassifier(max_iter=50, random_state=42))
])

# 3. Train the model
pipeline.fit(X, y)

# 4. Save the trained pipeline as a pickle file
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Machine Learning Model trained and saved successfully as 'fake_news_model.pkl'!")