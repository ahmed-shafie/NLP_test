import 'package:flutter/material.dart';

import 'screens/chat_screen.dart';
import 'services/settings_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final settings = await SettingsService.load();
  runApp(BankingNluApp(settings: settings));
}

class BankingNluApp extends StatelessWidget {
  const BankingNluApp({super.key, required this.settings});

  final SettingsService settings;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Banking NLU Assistant',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF7C3AED),
        useMaterial3: true,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        colorSchemeSeed: const Color(0xFF7C3AED),
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      home: ChatScreen(settings: settings),
    );
  }
}
