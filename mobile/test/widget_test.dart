import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:banking_nlu_assistant/main.dart';
import 'package:banking_nlu_assistant/services/settings_service.dart';

void main() {
  testWidgets('App renders the chat screen with the greeting', (tester) async {
    SharedPreferences.setMockInitialValues({});
    final settings = await SettingsService.load();

    await tester.pumpWidget(BankingNluApp(settings: settings));

    expect(find.text('Banking NLU Assistant'), findsOneWidget);
    expect(find.byIcon(Icons.mic), findsOneWidget);
    expect(find.byIcon(Icons.send), findsOneWidget);
  });
}
