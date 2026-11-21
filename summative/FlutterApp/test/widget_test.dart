import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:women_enterprise/main.dart';

void main() {
  testWidgets('Shows eight inputs and a Predict button', (tester) async {
    await tester.pumpWidget(const MyApp());

    expect(find.text('Business Sales Estimate'), findsOneWidget);
    expect(find.byType(TextField), findsNWidgets(8));
    expect(find.widgetWithText(FilledButton, 'Predict'), findsOneWidget);
    expect(find.text('Your sales estimate will appear here.'), findsOneWidget);
  });

  final invalidAgeCases = {
    '': 'Owner age (18–100) is required.',
    'abc': 'Owner age (18–100): enter a valid number.',
    '200': 'Owner age (18–100): enter a value between 18.0 and 100.0.',
  };

  for (final entry in invalidAgeCases.entries) {
    testWidgets('Shows an error for invalid age "${entry.key}"', (
      tester,
    ) async {
      await tester.pumpWidget(const MyApp());

      await tester.enterText(find.byType(TextField).first, entry.key);

      final button = find.widgetWithText(FilledButton, 'Predict');
      await tester.ensureVisible(button);
      await tester.tap(button);
      await tester.pumpAndSettle();

      expect(find.text(entry.value), findsOneWidget);
    });
  }
}
