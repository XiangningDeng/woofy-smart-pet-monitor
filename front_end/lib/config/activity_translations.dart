class ActivityTranslations {
  static Map<String, Map<String, String>> translations = {
    'en': {
      'Feeding': 'Feeding',
      'Playing': 'Playing',
      'Running': 'Running',
      'Sniffing': 'Sniffing',
      'Stilling': 'Stilling',
      'Walking': 'Walking',
    },
    'zh': {
      'Feeding': '进食',
      'Playing': '玩耍',
      'Running': '奔跑',
      'Sniffing': '嗅探',
      'Stilling': '静止',
      'Walking': '行走',
    },
  };

  static String getTranslation(String activity, String language) {
    return translations[language]?[activity] ?? activity;
  }
} 