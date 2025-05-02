import 'package:flutter/material.dart';

class SettingsService extends ChangeNotifier {
  bool _isDarkMode = false;
  String _currentLanguage = 'en';

  bool get isDarkMode => _isDarkMode;
  String get currentLanguage => _currentLanguage;

  void toggleTheme() {
    _isDarkMode = !_isDarkMode;
    notifyListeners();
  }

  void setLanguage(String language) {
    if (language != _currentLanguage) {
      _currentLanguage = language;
      notifyListeners();
    }
  }
} 