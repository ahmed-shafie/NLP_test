import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../models/conversation.dart';
import 'settings_service.dart';

/// Raised when the backend returns a non-2xx response.
class ApiException implements Exception {
  ApiException(this.message);
  final String message;
  @override
  String toString() => message;
}

/// Thin HTTP client for the Banking NLU conversation endpoints.
class ApiClient {
  ApiClient(this._settings);

  final SettingsService _settings;

  Uri _uri(String path) => Uri.parse('${_settings.baseUrl}$path');

  Map<String, String> _authHeaders() {
    final key = _settings.apiKey;
    return key.isEmpty ? {} : {'X-API-Key': key};
  }

  String? get _language =>
      _settings.language.isEmpty ? null : _settings.language;

  /// Send a text turn to `/conversation/text`.
  Future<ConversationResponse> sendText(String text, String? sessionId) async {
    final resp = await http.post(
      _uri('/conversation/text'),
      headers: {'content-type': 'application/json', ..._authHeaders()},
      body: jsonEncode({
        'text': text,
        'session_id': ?sessionId,
        'language': ?_language,
      }),
    );
    return _parse(resp.statusCode, resp.body);
  }

  /// Send a recorded audio clip to `/conversation/voice` (multipart upload).
  Future<ConversationResponse> sendVoice(
    String filePath,
    String? sessionId,
  ) async {
    final request = http.MultipartRequest('POST', _uri('/conversation/voice'))
      ..headers.addAll(_authHeaders())
      ..files.add(await http.MultipartFile.fromPath('audio', filePath));
    if (sessionId != null) request.fields['session_id'] = sessionId;
    if (_language != null) request.fields['language'] = _language!;

    final streamed = await request.send();
    final body = await streamed.stream.bytesToString();
    return _parse(streamed.statusCode, body);
  }

  /// Quick reachability check used by the Settings screen.
  Future<bool> ping() async {
    try {
      final resp = await http
          .get(_uri('/health'))
          .timeout(const Duration(seconds: 6));
      return resp.statusCode == 200;
    } on SocketException {
      return false;
    } catch (_) {
      return false;
    }
  }

  ConversationResponse _parse(int status, String body) {
    final Map<String, dynamic> json =
        body.isEmpty ? {} : jsonDecode(body) as Map<String, dynamic>;
    if (status >= 200 && status < 300) {
      return ConversationResponse.fromJson(json);
    }
    final detail = json['detail'] ??
        (json['error'] is Map ? (json['error'] as Map)['message'] : null);
    throw ApiException(detail?.toString() ?? 'Request failed (HTTP $status)');
  }
}
