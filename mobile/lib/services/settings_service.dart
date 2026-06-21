import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Persisted app settings: which backend to talk to and how to authenticate.
class SettingsService extends ChangeNotifier {
  SettingsService._(this._prefs);

  final SharedPreferences _prefs;

  static const _kBaseUrl = 'base_url';
  static const _kApiKey = 'api_key';
  static const _kLanguage = 'language';

  /// `10.0.2.2` is the Android emulator's alias for the host machine's
  /// `localhost`. Override it in Settings to point at a real server.
  static const defaultBaseUrl = 'http://10.0.2.2:8000';

  static Future<SettingsService> load() async {
    final prefs = await SharedPreferences.getInstance();
    return SettingsService._(prefs);
  }

  String get baseUrl =>
      (_prefs.getString(_kBaseUrl) ?? defaultBaseUrl).trim();

  String get apiKey => _prefs.getString(_kApiKey) ?? '';

  /// One of '', 'en', 'ar' — '' means auto-detect.
  String get language => _prefs.getString(_kLanguage) ?? '';

  Future<void> setBaseUrl(String value) async {
    await _prefs.setString(_kBaseUrl, value.trim());
    notifyListeners();
  }

  Future<void> setApiKey(String value) async {
    await _prefs.setString(_kApiKey, value.trim());
    notifyListeners();
  }

  Future<void> setLanguage(String value) async {
    await _prefs.setString(_kLanguage, value);
    notifyListeners();
  }
}
