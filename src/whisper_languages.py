import re


WHISPER_LANGUAGES = {
    'af': 'Afrikaans',
    'am': 'Amharic',
    'ar': 'Arabic',
    'as': 'Assamese',
    'az': 'Azerbaijani',
    'ba': 'Bashkir',
    'be': 'Belarusian',
    'bg': 'Bulgarian',
    'bn': 'Bengali',
    'bo': 'Tibetan',
    'br': 'Breton',
    'bs': 'Bosnian',
    'ca': 'Catalan',
    'cs': 'Czech',
    'cy': 'Welsh',
    'da': 'Danish',
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'es': 'Spanish',
    'et': 'Estonian',
    'eu': 'Basque',
    'fa': 'Persian',
    'fi': 'Finnish',
    'fo': 'Faroese',
    'fr': 'French',
    'gl': 'Galician',
    'gu': 'Gujarati',
    'ha': 'Hausa',
    'haw': 'Hawaiian',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hr': 'Croatian',
    'ht': 'Haitian Creole',
    'hu': 'Hungarian',
    'hy': 'Armenian',
    'id': 'Indonesian',
    'is': 'Icelandic',
    'it': 'Italian',
    'ja': 'Japanese',
    'jw': 'Javanese',
    'ka': 'Georgian',
    'kk': 'Kazakh',
    'km': 'Khmer',
    'kn': 'Kannada',
    'ko': 'Korean',
    'la': 'Latin',
    'lb': 'Luxembourgish',
    'ln': 'Lingala',
    'lo': 'Lao',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'mg': 'Malagasy',
    'mi': 'Maori',
    'mk': 'Macedonian',
    'ml': 'Malayalam',
    'mn': 'Mongolian',
    'mr': 'Marathi',
    'ms': 'Malay',
    'mt': 'Maltese',
    'my': 'Myanmar',
    'ne': 'Nepali',
    'nl': 'Dutch',
    'nn': 'Nynorsk',
    'no': 'Norwegian',
    'oc': 'Occitan',
    'pa': 'Punjabi',
    'pl': 'Polish',
    'ps': 'Pashto',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sa': 'Sanskrit',
    'sd': 'Sindhi',
    'si': 'Sinhala',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'sn': 'Shona',
    'so': 'Somali',
    'sq': 'Albanian',
    'sr': 'Serbian',
    'su': 'Sundanese',
    'sv': 'Swedish',
    'sw': 'Swahili',
    'ta': 'Tamil',
    'te': 'Telugu',
    'tg': 'Tajik',
    'th': 'Thai',
    'tk': 'Turkmen',
    'tl': 'Tagalog',
    'tr': 'Turkish',
    'tt': 'Tatar',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'uz': 'Uzbek',
    'vi': 'Vietnamese',
    'yi': 'Yiddish',
    'yo': 'Yoruba',
    'yue': 'Cantonese',
    'zh': 'Chinese',
}

WHISPER_LANGUAGE_CHOICES = [('Auto', 'auto')] + [
    (f"{name} ({code})", code)
    for code, name in sorted(WHISPER_LANGUAGES.items(), key=lambda item: item[1])
]

WHISPER_LANGUAGE_NAMES = {
    name.lower(): code for code, name in WHISPER_LANGUAGES.items()
}

LANGUAGE_CODE_PATTERN = re.compile(r'\(([a-z]{2,3})\)\s*$', re.IGNORECASE)
AUTO_LANGUAGE_VALUES = {'auto', 'automatic', 'auto detect', 'autodetect'}


def normalize_whisper_language(value):
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    normalized = value.strip()
    if not normalized:
        return None

    lowered = normalized.lower()
    if lowered in AUTO_LANGUAGE_VALUES:
        return None

    if lowered in WHISPER_LANGUAGES:
        return lowered

    if lowered in WHISPER_LANGUAGE_NAMES:
        return WHISPER_LANGUAGE_NAMES[lowered]

    match = LANGUAGE_CODE_PATTERN.search(normalized)
    if match:
        return match.group(1).lower()

    return lowered