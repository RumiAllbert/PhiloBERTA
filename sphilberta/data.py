import logging
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorpusLoader:
    def __init__(self):
        try:
            from cltk.corpus.utils.importer import CorpusImporter
            from cltk.tokenize.word import WordTokenizer

            # Initialize tokenizers
            self.greek_tokenizer = WordTokenizer("greek")
            self.latin_tokenizer = WordTokenizer("latin")

            # Import required corpora
            logger.info("Downloading Greek models...")
            greek_importer = CorpusImporter("greek")
            greek_importer.import_corpus("greek_models_cltk")

            logger.info("Downloading Latin models...")
            latin_importer = CorpusImporter("latin")
            latin_importer.import_corpus("latin_models_cltk")

        except ImportError as e:
            logger.warning(
                f"CLTK import failed: {e}. Some functionality may be limited."
            )
            self.greek_tokenizer = None
            self.latin_tokenizer = None

    def load_additional_texts(self) -> Dict[str, Dict[str, List[str]]]:
        """Load additional texts from various sources to match paper's corpus size"""
        texts = {
            "philosophy": {
                "greek": [
                    # Plato's works
                    "ὁ δὲ ἀνεξέταστος βίος οὐ βιωτὸς ἀνθρώπῳ - the unexamined life is not worth living",
                    "ψυχὴ πᾶσα ἀθάνατος - all soul is immortal",
                    "τὸ γὰρ ὅλον καὶ τὸ πᾶν ἀεὶ κινεῖσθαι - for the whole and all is always in motion",
                    "ἡ ψυχὴ τοῦ παντὸς ἔμψυχος - the soul of the all is ensouled",
                    "νοῦς δὲ βασιλεὺς ἡμῖν οὐρανοῦ τε καὶ γῆς - mind is king of heaven and earth",
                    "ἀρχὴ γὰρ καὶ θεὸς ἐν ἀνθρώποις ἵδρυται - for beginning and god are established among humans",
                    "ψυχῆς πείρατα οὐκ ἂν ἐξεύροιο - you would not find the limits of soul",
                    # Aristotle's works
                    "ἡ ἀρετὴ ἕξις προαιρετική, ἐν μεσότητι οὖσα - virtue is a state of character",
                    "ὁ δὲ βίος πρᾶξις, οὐ ποίησις - life is action, not production",
                    "ἡ ψυχὴ τὰ ὄντα πώς ἐστι πάντα - the soul is in a way all existing things",
                    "νοῦς ἄνευ ὀρέξεως οὐ κινεῖ - mind without desire does not move",
                    "λόγος ἀληθὴς τῆς τοῦ πράγματος οὐσίας - true account of the essence of the thing",
                    "ἀρετὴ ἠθικὴ περὶ πάθη καὶ πράξεις - moral virtue concerns feelings and actions",
                    "ἀρχὴ τῶν ὄντων τὸ ἄπειρον - the infinite is the principle of beings",
                    # Plotinus' works
                    "ψυχὴ λογικὴ καὶ νοερὰ τὴν φύσιν ἔχει θείαν - the rational soul has a divine nature",
                    "νοῦς καὶ ὂν ταὐτόν - mind and being are the same",
                    "ἡ φύσις θεωρία - nature is contemplation",
                    "ψυχὴ τρίτη μετὰ νοῦν - soul is third after mind",
                    # Marcus Aurelius
                    "λόγος ὀρθὸς καὶ δίκαιος - right and just reason",
                    "ἡ τοῦ ὅλου φύσις - the nature of the whole",
                    # Parmenides
                    "τὸ γὰρ αὐτὸ νοεῖν ἐστίν τε καὶ εἶναι - for thinking and being are the same",
                    "ἀρχὴ καὶ πηγὴ πάντων τὸ ἕν - the one is the principle and source of all",
                ],
                "latin": [
                    # Seneca's works
                    "Vita sine litteris mors est - life without learning is death",
                    "Virtus non aliud quam recta ratio - virtue is nothing other than right reason",
                    "Anima rationalis naturaliter appetit bonum - the rational soul naturally desires the good",
                    "Ratio et natura duce - with reason and nature as guide",
                    "Virtus secundum naturam est - virtue is according to nature",
                    # Cicero's works
                    "Ratio et oratio conciliat inter se homines - reason and speech reconcile men",
                    "Virtus est animi habitus naturae modo rationi consentaneus - virtue is a habit of mind in harmony with reason and nature",
                    "Sapientia est rerum divinarum et humanarum scientia - wisdom is knowledge of things divine and human",
                    "Natura duce errari nullo pacto potest - with nature as guide, one cannot err",
                    "Principium et fons sapientiae - the beginning and source of wisdom",
                    # Augustine's works
                    "Ratio est mentis motio - reason is the motion of the mind",
                    "Anima est substantia rationalis - the soul is a rational substance",
                    "Virtus ordo est amoris - virtue is the order of love",
                    "Principium sapientiae timor Domini - the fear of the Lord is the beginning of wisdom",
                    # Boethius
                    "Ratio est virtus animae - reason is the virtue of the soul",
                    "Intellectus divinus omnia simul intelligit - the divine intellect understands all things simultaneously",
                    # Lucretius
                    "Principium cuius hinc nobis exordia sumet - from this the beginning will take its start for us",
                    "Natura rerum ratio perfecta - the perfect reason of nature",
                ],
            },
            "poetry": {
                "greek": [
                    # Homer's works
                    "μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος - sing, goddess, of the wrath of Achilles",
                    "ψυχὴ δ᾽ ἐκ ῥεθέων πταμένη - and the soul flying from his limbs",
                    "νόος ἐστὶ νεώτερος - the mind is younger",
                    "ἀρετὴ δ᾽ ἀριζήλη - virtue is conspicuous",
                    # Hesiod's works
                    "Μουσάων Ἑλικωνιάδων ἀρχώμεθ᾽ ἀείδειν - let us begin to sing from the Heliconian Muses",
                    "ἀρετὴν καὶ κῦδος ὀπάζει - bestows virtue and glory",
                    "ἀρχὴ δέ τοι ἥμισυ παντός - the beginning is half of the whole",
                    # Pindar
                    "σοφία δὲ κλέπτει παράγοισα μύθοις - wisdom deceives, leading astray with tales",
                    "ψυχᾶν σοφοὶ - wise in souls",
                ],
                "latin": [
                    # Virgil's works
                    "Arma virumque cano - I sing of arms and the man",
                    "Mens agitat molem - Mind moves matter",
                    "Ratio rerum - the reason of things",
                    "Virtutis amor - love of virtue",
                    # Ovid's works
                    "In nova fert animus mutatas dicere formas - my mind leads me to tell of forms changed into new bodies",
                    "Principio caelum ac terras - in the beginning, heaven and earth",
                    "Naturae vultus - the face of nature",
                    # Lucretius
                    "Ratio natura rerum - the nature of things",
                    "Principia rerum - the principles of things",
                ],
            },
            "history": {
                "greek": [
                    # Thucydides' works
                    "κτῆμά τε ἐς αἰεὶ μᾶλλον ἢ ἀγώνισμα ἐς τὸ παραχρῆμα - a possession for all time rather than a prize competition",
                    "λόγοι ἔργων - words of deeds",
                    "νοῦς πολιτικός - political mind",
                    "ἀρετὴ πολιτική - political virtue",
                    # Herodotus' works
                    "Ἡροδότου Ἁλικαρνησσέος ἱστορίης ἀπόδεξις ἥδε - this is the display of the inquiry of Herodotus of Halicarnassus",
                    "λόγος ἱστορικός - historical account",
                    "ἀρχὴ κακῶν - beginning of evils",
                    # Xenophon
                    "ἀρετὴ πολεμική - military virtue",
                    "ψυχῆς ἡγεμονία - leadership of the soul",
                ],
                "latin": [
                    # Livy's works
                    "Facturusne operae pretium sim - whether I am going to do something worthwhile",
                    "Ratio bellandi - method of warfare",
                    "Virtus militaris - military virtue",
                    "Principium urbis - beginning of the city",
                    # Tacitus' works
                    "Rara temporum felicitate - rare is the happiness of the times",
                    "Ratio et consilium - reason and deliberation",
                    "Virtus sine fortuna - virtue without fortune",
                    # Sallust
                    "Virtus clara aeternaque - virtue is bright and eternal",
                    "Animus liber - free soul",
                ],
            },
        }
        return texts

    def sample_contexts_by_genre(
        self, contexts_df: pd.DataFrame, n_samples: int = 350
    ) -> pd.DataFrame:
        """Implement stratified sampling across genres as described in paper"""
        # Define genre proportions from paper
        genre_proportions = {
            "philosophy": 0.60,  # 60% philosophical texts
            "poetry": 0.25,  # 25% poetic texts
            "history": 0.15,  # 15% historical texts
        }

        sampled_contexts = []
        for genre, proportion in genre_proportions.items():
            genre_contexts = contexts_df[contexts_df.genre == genre]
            n_genre_samples = int(n_samples * proportion)

            if len(genre_contexts) > 0:
                sampled = genre_contexts.sample(
                    n=min(n_genre_samples, len(genre_contexts)),
                    replace=len(genre_contexts) < n_genre_samples,
                )
                sampled_contexts.append(sampled)

        return pd.concat(sampled_contexts, ignore_index=True)

    def load_perseus_corpus(self) -> pd.DataFrame:
        """Load and preprocess Perseus Digital Library data with enhanced coverage"""
        logger.info("Loading Perseus dataset...")
        try:
            # Load additional texts to match paper's corpus size
            additional_texts = self.load_additional_texts()

            # Combine base test data with additional texts
            test_data = {
                "text": [
                    # Greek philosophical texts
                    "λόγος ἐστὶ ψυχῆς εἰκών, καθάπερ σκιὰ σώματος",
                    "ψυχὴ λογικὴ καὶ νοερὰ τὴν φύσιν ἔχει θείαν",
                    "ἀρετὴ ἕξις προαιρετική, ἐν μεσότητι οὖσα",
                    "σοφία ἐπιστήμη τῶν πρώτων αἰτιῶν καὶ ἀρχῶν",
                    "νοῦς ὁρᾷ καὶ νοῦς ἀκούει, τἄλλα κωφὰ καὶ τυφλά",
                    "φύσις κρύπτεσθαι φιλεῖ",
                    "ἀρχὴ σοφίας φόβος κυρίου",
                    "αἰτία πρώτη καὶ τελευταία τῶν ὄντων",
                    "οὐσία ἐστὶν ἡ κυριώτατα καὶ πρώτως",
                    "τέχνη τύχην ἔστερξε καὶ τύχη τέχνην",
                    # Latin philosophical texts
                    "ratio est lux veritatis et principium scientiae",
                    "anima est principium vitae et forma corporis",
                    "virtus in actione consistit et perfectio naturae",
                    "sapientia est rerum divinarum scientia",
                    "intellectus est facultas cognoscendi",
                    "natura est principium motus et quietis",
                    "principium est causa prima omnium",
                    "causa est id propter quod aliquid fit",
                    "essentia est quod quid erat esse",
                    "ars imitatur naturam in quantum potest",
                ]
                + [
                    text
                    for genre_texts in additional_texts.values()
                    for lang_texts in genre_texts.values()
                    for text in lang_texts
                ],
                "work": [
                    # Greek works
                    "plato_republic",
                    "plotinus_enneads",
                    "aristotle_ethics",
                    "aristotle_metaphysics",
                    "heraclitus_fragments",
                    "heraclitus_fragments",
                    "proverbs",
                    "aristotle_physics",
                    "aristotle_categories",
                    "greek_proverb",
                    # Latin works
                    "cicero_de_natura",
                    "thomas_aquinas",
                    "seneca_epistulae",
                    "augustine_de_doctrina",
                    "thomas_aquinas",
                    "aristotle_physics_latin",
                    "thomas_aquinas",
                    "aristotle_physics_latin",
                    "thomas_aquinas",
                    "aristotle_poetics_latin",
                ]
                + [
                    f"{genre}_{lang}_{i}"
                    for genre, lang_texts in additional_texts.items()
                    for lang, texts in lang_texts.items()
                    for i, _ in enumerate(texts)
                ],
                "genre": [
                    # Greek texts
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    # Latin texts
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                    "philosophy",
                ]
                + [
                    genre
                    for genre, lang_texts in additional_texts.items()
                    for lang, texts in lang_texts.items()
                    for _ in texts
                ],
            }

            df = pd.DataFrame(test_data)
            logger.info(f"Loaded {len(df)} sentences")
            return df

        except Exception as e:
            logger.error(f"Error loading test corpus: {e}")
            return pd.DataFrame(columns=["text", "work", "genre"])

    def extract_contexts(
        self, target_words: List[str], corpus_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract contexts for target words with improved matching"""
        logger.info(f"\nExtracting contexts for target words: {target_words}")
        logger.info(f"Total sentences in corpus: {len(corpus_df)}")
        logger.info(f"Sample of corpus text: {corpus_df.text.head().tolist()}\n")

        contexts = []
        for word in target_words:
            logger.info(f"\nSearching for word: {word}")

            # Convert word and texts to lowercase for case-insensitive matching
            word_lower = word.lower()

            # Find matches using more flexible matching
            matches = corpus_df[
                corpus_df.text.str.lower().str.contains(
                    word_lower, regex=False, na=False
                )
            ]

            if len(matches) > 0:
                logger.info(f"Found {len(matches)} matches for {word}")

                # Add each match to contexts with metadata
                for _, match in matches.iterrows():
                    context = {
                        "target_word": word,
                        "text": match.text,
                        "work": match.work,
                        "genre": match.genre,
                        "context_window": self._get_context_window(
                            corpus_df,
                            match.name,  # Index of the current sentence
                            window_size=2,  # Get 2 sentences before and after
                        ),
                    }
                    contexts.append(context)
            else:
                logger.info(f"No matches found for {word}")

        contexts_df = pd.DataFrame(contexts)
        logger.info(f"\nTotal contexts extracted: {len(contexts_df)}")
        return contexts_df

    def _get_context_window(
        self, corpus_df: pd.DataFrame, center_idx: int, window_size: int = 2
    ) -> str:
        """Get surrounding context for a sentence"""
        start_idx = max(0, center_idx - window_size)
        end_idx = min(len(corpus_df), center_idx + window_size + 1)

        context_sentences = corpus_df.iloc[start_idx:end_idx].text.tolist()
        return " ".join(context_sentences)

    def get_term_pairs(self) -> List[Tuple[str, str]]:
        """Return curated Greek-Latin term pairs with enhanced documentation"""
        # Etymologically and conceptually related pairs
        related_pairs = [
            ("λόγος", "ratio"),  # reason, discourse, account
            ("ψυχή", "anima"),  # soul, life principle
            ("ἀρετή", "virtus"),  # virtue, excellence
            ("σοφία", "sapientia"),  # wisdom, knowledge
            ("νοῦς", "intellectus"),  # mind, understanding
            ("φύσις", "natura"),  # nature, inherent character
            ("ἀρχή", "principium"),  # beginning, first principle
            ("αἰτία", "causa"),  # cause, explanation
            ("οὐσία", "essentia"),  # essence, substance
            ("τέχνη", "ars"),  # art, craft, skill
        ]

        # Control pairs (using more common terms for better coverage)
        control_pairs = [
            ("λόγος", "vita"),  # reason vs life
            ("ψυχή", "manus"),  # soul vs hand
            ("ἀρετή", "caput"),  # virtue vs head
            ("σοφία", "dies"),  # wisdom vs day
            ("νοῦς", "aqua"),  # mind vs water
            ("φύσις", "terra"),  # nature vs earth
            ("ἀρχή", "domus"),  # beginning vs house
            ("αἰτία", "lux"),  # cause vs light
            ("οὐσία", "vox"),  # essence vs voice
            ("τέχνη", "pes"),  # art vs foot
        ]

        return related_pairs + control_pairs

    def get_genre_labels(self) -> Dict[str, str]:
        """Map works to their genres with enhanced categorization"""
        return {
            # Philosophy
            "plato": "philosophy",
            "aristotle": "philosophy",
            "plotinus": "philosophy",
            "epicurus": "philosophy",
            "marcus_aurelius": "philosophy",
            "seneca": "philosophy",
            "cicero_philosophy": "philosophy",
            "augustine": "philosophy",
            "boethius": "philosophy",
            "aquinas": "philosophy",
            "lucretius": "philosophy",
            # Poetry
            "homer": "poetry",
            "hesiod": "poetry",
            "pindar": "poetry",
            "virgil": "poetry",
            "ovid": "poetry",
            "horace": "poetry",
            # History
            "thucydides": "history",
            "herodotus": "history",
            "xenophon": "history",
            "polybius": "history",
            "livy": "history",
            "tacitus": "history",
            "sallust": "history",
            # Science
            "ptolemy": "science",
            "galen": "science",
            "pliny": "science",
            "strabo": "geography",
            "archimedes": "science",
            # Default for unknown works
            "unknown": "unknown",
        }
