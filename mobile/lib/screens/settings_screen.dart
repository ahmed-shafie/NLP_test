import 'package:flutter/material.dart';

import '../services/api_client.dart';
import '../services/settings_service.dart';

/// Configure the backend base URL, optional API key, and language.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key, required this.settings});

  final SettingsService settings;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _baseUrl;
  late final TextEditingController _apiKey;
  late String _language;
  String? _pingResult;
  bool _pinging = false;

  @override
  void initState() {
    super.initState();
    _baseUrl = TextEditingController(text: widget.settings.baseUrl);
    _apiKey = TextEditingController(text: widget.settings.apiKey);
    _language = widget.settings.language;
  }

  @override
  void dispose() {
    _baseUrl.dispose();
    _apiKey.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    await widget.settings.setBaseUrl(_baseUrl.text);
    await widget.settings.setApiKey(_apiKey.text);
    await widget.settings.setLanguage(_language);
    if (mounted) Navigator.of(context).pop();
  }

  Future<void> _test() async {
    setState(() {
      _pinging = true;
      _pingResult = null;
    });
    // Save the URL first so the ping uses the edited value.
    await widget.settings.setBaseUrl(_baseUrl.text);
    await widget.settings.setApiKey(_apiKey.text);
    final ok = await ApiClient(widget.settings).ping();
    if (!mounted) return;
    setState(() {
      _pinging = false;
      _pingResult = ok ? 'Connected — backend is reachable.' : 'Could not reach the backend.';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          TextField(
            controller: _baseUrl,
            keyboardType: TextInputType.url,
            autocorrect: false,
            decoration: const InputDecoration(
              labelText: 'Backend base URL',
              hintText: SettingsService.defaultBaseUrl,
              helperText:
                  'Android emulator → http://10.0.2.2:8000 ; real device → your server URL',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _apiKey,
            autocorrect: false,
            obscureText: true,
            decoration: const InputDecoration(
              labelText: 'API key (only if auth is enabled)',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          DropdownButtonFormField<String>(
            initialValue: _language,
            decoration: const InputDecoration(
              labelText: 'Language',
              border: OutlineInputBorder(),
            ),
            items: const [
              DropdownMenuItem(value: '', child: Text('Auto-detect')),
              DropdownMenuItem(value: 'en', child: Text('English')),
              DropdownMenuItem(value: 'ar', child: Text('العربية')),
            ],
            onChanged: (v) => setState(() => _language = v ?? ''),
          ),
          const SizedBox(height: 24),
          OutlinedButton.icon(
            onPressed: _pinging ? null : _test,
            icon: _pinging
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.wifi_tethering),
            label: const Text('Test connection'),
          ),
          if (_pingResult != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                _pingResult!,
                style: TextStyle(
                  color: _pingResult!.startsWith('Connected')
                      ? Colors.green.shade700
                      : Theme.of(context).colorScheme.error,
                ),
              ),
            ),
          const SizedBox(height: 24),
          FilledButton(onPressed: _save, child: const Text('Save')),
        ],
      ),
    );
  }
}
