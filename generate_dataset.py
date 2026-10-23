#!/usr/bin/env python3
"""Generate a small, fully synthetic noun/verb/other teaching dataset; standard library only."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOKEN_PATTERN = r"[A-Za-z]+(?:'[A-Za-z]+)?"

# target | three noun predicates | three noun-object verbs | noun adjective
NOUN_DATA = """
watch|tells the time|needs a battery|belongs to Mia|fixed|bought|checked|expensive
play|ends at nine|has three acts|starts soon|enjoyed|wrote|liked|funny
call|lasted ten minutes|can wait|came from Dad|missed|received|answered|brief
walk|takes an hour|can be relaxing|ends at the lake|enjoyed|planned|finished|long
run|starts at noon|can be tiring|ends near home|finished|planned|enjoyed|long
drink|tastes sweet|can cool you down|contains lemon|spilled|ordered|finished|cold
kick|sent the ball flying|can hurt|missed the goal|blocked|practiced|noticed|powerful
park|closes at dusk|has tall trees|contains a pond|visited|found|photographed|peaceful
plant|needs more water|has green leaves|belongs to Leo|watered|bought|moved|healthy
dance|ends at midnight|has simple steps|starts after dinner|enjoyed|learned|practiced|lively
brush|has soft bristles|needs cleaning|belongs to Sam|cleaned|bought|dropped|soft
wave|looked friendly|can knock you over|splashed the boat|saw|photographed|noticed|huge
smile|made me happy|can show kindness|looked friendly|noticed|returned|remembered|warm
answer|seems correct|can be surprising|needs more detail|checked|wrote|changed|clear
dream|felt very real|can seem strange|involved a dragon|remembered|described|shared|strange
light|fills the room|can help us read|comes from a lamp|noticed|installed|adjusted|bright
""".strip()

# Ten short complements per target: every slot uses the target as a base-form verb.
VERB_DATA = {
    "watch": ["the birds", "the boats", "the road", "the stars", "the game", "the baby", "the sunset", "scary films", "the match", "the parade"],
    "play": ["chess", "outside", "the piano", "cards", "tennis", "together", "the drums", "loud music", "football", "in the garden"],
    "call": ["our friends", "home", "your sister", "me tomorrow", "a taxi", "the doctor", "my brother", "strangers", "your parents", "for help"],
    "walk": ["to school", "along the river", "beside me", "this way", "home together", "on the path", "to the beach", "on the grass", "to work", "through town"],
    "run": ["around the track", "every morning", "to the gate", "up the hill", "very fast", "for the bus", "a small shop", "in the hallway", "after school", "toward the finish"],
    "drink": ["water", "tea", "some juice", "slowly", "from this cup", "enough water", "hot chocolate", "coffee at night", "milk", "after exercise"],
    "kick": ["the ball", "with both feet", "toward the goal", "gently", "a ball outside", "with care", "a ball around", "the furniture", "with your left foot", "during practice"],
    "park": ["near the school", "beside the shop", "behind the house", "over there", "by the gate", "inside the garage", "near the lake", "on the bridge", "near home", "beside the road"],
    "plant": ["beans", "flowers", "these seeds", "a tree here", "carrots", "the bulbs today", "roses", "trees in winter", "vegetables", "in spring"],
    "dance": ["together", "after dinner", "with me", "to this song", "all evening", "on the stage", "at the party", "in the kitchen", "on weekends", "at the festival"],
    "brush": ["their teeth", "the dog's fur", "your hair", "the dust away", "the horse", "your coat", "my hair", "the cat", "your teeth", "the pony"],
    "wave": ["at the driver", "from the window", "to your aunt", "both hands", "at the crowd", "from the boat", "at my friend", "at strangers", "to the neighbors", "goodbye"],
    "smile": ["at our friends", "for the camera", "at the baby", "for a photo", "at each other", "politely", "at my neighbor", "during sad films", "when you arrive", "at the visitors"],
    "answer": ["the questions", "the phone", "me honestly", "this question", "the door", "clearly", "your question", "unknown callers", "the teacher", "in class"],
    "dream": ["about flying", "every night", "of a better world", "big", "about the future", "of better days", "of visiting Mars", "about work", "in color", "about dragons"],
    "light": ["the candles", "a small fire", "the lamp", "this candle", "the stove", "a lantern", "the fireplace", "fires indoors", "the torch", "the campfire"],
}

AFTER = {
    "watch": "we stay quiet", "play": "we laugh together",
    "call": "someone answers", "walk": "we notice new shops",
    "run": "our friends cheer", "drink": "we feel refreshed",
    "kick": "the coach gives advice", "park": "we check the signs",
    "plant": "the soil feels warm", "dance": "the music sounds louder",
    "brush": "the pony stands still", "wave": "our friends greet us",
    "smile": "they seem happy", "answer": "the teacher listens",
    "dream": "anything seems possible", "light": "the wood begins to glow",
}

# Broaden the target vocabulary beyond words that can be both nouns and verbs.
# For these examples, contexts (rather than template families) are held out.
NOUN_TARGETS = ("birds", "boats", "dogs", "cats", "trees", "books", "teachers",
                "students", "flowers", "stars", "friends", "cars", "houses",
                "horses", "lamps", "chairs")
OTHER_PATTERNS = {
    "they": "They watch the {plural}.",
    "we": "We watch the {plural}.",
    "you": "You watch the {plural}.",
    "i": "I watch the {plural}.",
    "the": "They watch the {plural}.",
    "a": "They see a {singular}.",
    "this": "They see this {singular}.",
    "my": "They see my {singular}.",
    "your": "They see your {singular}.",
    "to": "They walk to the {place}.",
    "with": "They play with the {singular}.",
    "and": "They watch the {plural} and smile.",
    "quickly": "They quickly notice the {plural}.",
    "very": "The {plural} are very quiet.",
    "happy": "The happy {plural} play outside.",
    "not": "They do not watch the {plural}.",
    "please": "Please watch the {plural}.",
}
PLURALS = ("birds", "dogs", "cats", "horses", "students", "friends",
           "teachers", "boats", "cars", "flowers")
SINGULARS = ("bird", "dog", "cat", "horse", "student", "friend",
             "teacher", "child", "puppy", "kitten")
PLACES = ("park", "school", "lake", "shop", "house", "garden",
          "beach", "river", "station", "bridge")


def build():
    rows = []
    for line in NOUN_DATA.splitlines():
        word, p1, p2, p3, a1, a2, a3, adjective = line.split("|")
        article = "an" if word[0] in "aeiou" else "a"
        adjective_article = "an" if adjective[0] in "aeiou" else "a"
        nouns = [
            (f"The {word} {p1}.", "determiner_noun_subject"),
            (f"{article.capitalize()} {word} {p2}.", "determiner_noun_subject"),
            (f"This {word} {p3}.", "determiner_noun_subject"),
            (f"I {a1} the {word}.", "noun_direct_object"),
            (f"We {a2} {article} {word}.", "noun_direct_object"),
            (f"She {a3} this {word}.", "noun_direct_object"),
            (f"Alex's {word} {p1}.", "possessive_noun_subject"),
            (f"We talked about the {word}.", "noun_prepositional_object"),
            (f"Was that {word} {adjective}?", "noun_subject_question"),
            (f"What {adjective_article} {adjective} {word}!", "noun_exclamation"),
        ]
        c = VERB_DATA[word]
        verbs = [
            (f"They {word} {c[0]}.", "pronoun_verb_declarative"),
            (f"We {word} {c[1]}.", "pronoun_verb_declarative"),
            (f"Please {word} {c[2]}.", "verb_imperative"),
            (f"{word.capitalize()} {c[3]}!", "verb_imperative"),
            (f"We can {word} {c[4]}.", "verb_modal"),
            (f"You must {word} {c[5]}.", "verb_modal"),
            (f"I want to {word} {c[6]}.", "verb_infinitive"),
            (f"We do not {word} {c[7]}.", "verb_negation"),
            (f"Do you {word} {c[8]}?", "verb_question"),
            (f"When we {word} {c[9]}, {AFTER[word]}.", "verb_subordinate_clause"),
        ]
        for label, examples in [("noun", nouns), ("verb", verbs)]:
            for i, (sentence, family) in enumerate(examples):
                split = "train" if i < 6 else "validation" if i < 8 else "test"
                tokens = re.findall(TOKEN_PATTERN, sentence)
                matches = [j for j, token in enumerate(tokens) if token.lower() == word]
                assert len(matches) == 1
                rows.append(dict(id=f"{word}-{label}-{i+1:02d}", sentence=sentence,
                                 target=word, target_index=matches[0], label=label,
                                 split=split, template_family=family))
    noun_patterns = [
        "They watch the {word}.", "The {word} are outside.",
        "We see the {word}.", "You notice these {word}.",
        "I like the {word}.", "She likes those {word}.",
        "Can you see the {word}?", "We talked about the {word}.",
        "Where are the {word}?", "Look at those beautiful {word}!",
    ]
    for word in NOUN_TARGETS:
        for i, pattern in enumerate(noun_patterns):
            sentence = pattern.format(word=word)
            tokens = re.findall(TOKEN_PATTERN, sentence.lower())
            split = "train" if i < 6 else "validation" if i < 8 else "test"
            rows.append(dict(id=f"{word}-noun-{i+1:02d}", sentence=sentence,
                             target=word, target_index=tokens.index(word), label="noun",
                             split=split, template_family=f"broad_noun_{i+1:02d}"))
    for word, pattern in OTHER_PATTERNS.items():
        for i in range(10):
            sentence = pattern.format(plural=PLURALS[i], singular=SINGULARS[i],
                                      place=PLACES[i])
            if i >= 6:
                sentence = sentence[:-1] + (" today." if i < 8 else " again.")
            tokens = re.findall(TOKEN_PATTERN, sentence.lower())
            split = "train" if i < 6 else "validation" if i < 8 else "test"
            rows.append(dict(id=f"{word}-other-{i+1:02d}", sentence=sentence,
                             target=word, target_index=tokens.index(word), label="other",
                             split=split, template_family="other_" + word))
    return rows

def audit(rows):
    assert len(rows) == 320 + 10 * (len(NOUN_TARGETS) + len(OTHER_PATTERNS))
    assert len({r["id"] for r in rows}) == len(rows)
    # A sentence can annotate multiple target positions, but a supervised
    # sentence/position pair must be unique and stay in one split.
    assert len({(r["sentence"].lower(), r["target_index"]) for r in rows}) == len(rows)
    expected = {"train": 6, "validation": 2, "test": 2}
    counts = Counter((r["target"], r["label"], r["split"]) for r in rows)
    for word in VERB_DATA:
        for label in ("noun", "verb"):
            for split, n in expected.items():
                assert counts[word, label, split] == n
    for targets, label in [(NOUN_TARGETS, "noun"), (OTHER_PATTERNS, "other")]:
        for word in targets:
            for split, n in expected.items():
                assert counts[word, label, split] == n
    families, sentence_splits = {}, {}
    for row in rows:
        tokens = re.findall(TOKEN_PATTERN, row["sentence"])
        assert tokens[row["target_index"]].lower() == row["target"]
        assert row["label"] in ("noun", "verb", "other")
        families.setdefault(row["template_family"], set()).add(row["split"])
        sentence_splits.setdefault(row["sentence"].lower(), set()).add(row["split"])
    assert all(len(splits) == 1 for splits in sentence_splits.values())
    assert all(len(splits) == 1 for family, splits in families.items()
               if not family.startswith("other_"))
    return {"examples": len(rows), "targets": sorted({r["target"] for r in rows}),
            "labels": ["noun", "verb", "other"],
            "counts_by_split": dict(Counter(r["split"] for r in rows)),
            "counts_by_label": dict(Counter(r["label"] for r in rows)),
            "template_families_by_split": {
                split: sorted(f for f, splits in families.items() if split in splits)
                for split in expected},
            "token_pattern": TOKEN_PATTERN, "target_index_base": 0,
            "source": "Hand-authored synthetic examples expanded by deterministic templates",
            "split_strategy": "Noun/verb templates are disjoint across splits. Other-label targets share templates with held-out context words. Full sentences never cross splits.",
            "checks": ["unique IDs and sentence/target pairs", "6/2/2 rows per target and label",
                       "valid target indices", "no sentence overlap across splits"]}

if __name__ == "__main__":
    dataset = build()
    manifest = audit(dataset)
    for split in ("train", "validation", "test"):
        with (ROOT / f"{split}.jsonl").open("w") as out:
            for row in dataset:
                if row["split"] == split:
                    out.write(json.dumps(row) + "\n")
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts_by_split"]))
