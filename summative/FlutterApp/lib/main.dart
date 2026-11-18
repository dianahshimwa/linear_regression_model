import 'package:flutter/material.dart';

import 'api_service.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Business Sales Estimate',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF526B65),
        ),
        scaffoldBackgroundColor: const Color(0xFFFAFAFA),
        useMaterial3: true,
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final controllers = {
    'owner_age': TextEditingController(text: '44'),
    'employees': TextEditingController(text: '0'),
    'hours_worked_last_week': TextEditingController(text: '64'),
    'education_level': TextEditingController(text: 'level_12'),
    'county': TextEditingController(text: 'KISII'),
    'business_type':
        TextEditingController(text: 'GRAIN SELLER AND GROCERY'),
    'advertises': TextEditingController(text: 'no'),
    'keeps_written_records': TextEditingController(text: 'no'),
  };

  final labels = {
    'owner_age': 'Owner age (18–100)',
    'employees': 'Number of employees (0–3)',
    'hours_worked_last_week': 'Hours worked last week (0–168)',
    'education_level': 'Education code (level_0 to level_18)',
    'county': 'County (KITUI, KISII, KAKAMEGA, EMBU)',
    'business_type': 'Business type',
    'advertises': 'Advertised in last six months? (yes/no)',
    'keeps_written_records': 'Keeps written records? (yes/no)',
  };

  bool loading = false;
  bool hasError = false;
  String result = 'Your sales estimate will appear here.';

  String text(String name) {
    return controllers[name]!.text.trim();
  }

  double readNumber(String name, double minimum, double maximum) {
    final value = double.tryParse(text(name));

    if (value == null || !value.isFinite) {
      throw FormatException(
        '${labels[name]}: enter a valid number.',
      );
    }

    if (value < minimum || value > maximum) {
      throw FormatException(
        '${labels[name]}: enter a value between $minimum and $maximum.',
      );
    }

    return value;
  }

  Map<String, dynamic> collectInputs() {
    for (final name in controllers.keys) {
      if (text(name).isEmpty) {
        throw FormatException('${labels[name]} is required.');
      }
    }

    final age = readNumber('owner_age', 18, 100);
    final hours = readNumber('hours_worked_last_week', 0, 168);
    final employees = int.tryParse(text('employees'));

    if (employees == null || employees < 0 || employees > 3) {
      throw const FormatException(
        'Employees must be a whole number between 0 and 3.',
      );
    }

    final education = text('education_level').toLowerCase();

    if (!RegExp(r'^level_([0-9]|1[0-8])$').hasMatch(education)) {
      throw const FormatException(
        'Enter an education code from level_0 to level_18.',
      );
    }

    final county = text('county').toUpperCase();

    if (!['KITUI', 'KISII', 'KAKAMEGA', 'EMBU'].contains(county)) {
      throw const FormatException(
        'County must be KITUI, KISII, KAKAMEGA, or EMBU.',
      );
    }

    final advertises = text('advertises').toLowerCase();
    final records = text('keeps_written_records').toLowerCase();

    if (!['yes', 'no'].contains(advertises) ||
        !['yes', 'no'].contains(records)) {
      throw const FormatException(
        'Enter yes or no for advertising and written records.',
      );
    }

    return {
      'owner_age': age,
      'employees': employees,
      'hours_worked_last_week': hours,
      'education_level': education,
      'county': county,
      'business_type': text('business_type').toUpperCase(),
      'advertises': advertises,
      'keeps_written_records': records,
    };
  }

  Future<void> predict() async {
    FocusScope.of(context).unfocus();

    setState(() {
      loading = true;
      hasError = false;
      result =
          'Getting estimate. The server may take a moment to wake up.';
    });

    String message;
    bool failed = true;

    try {
      final inputs = collectInputs();
      final prediction = await ApiService.predict(inputs);

      message =
          'Estimated weekly sales: ${prediction.toStringAsFixed(2)} KSh';
      failed = false;
    } on FormatException catch (error) {
      message = error.message;
    } catch (_) {
      message = 'Could not get an estimate. Please try again.';
    }

    if (!mounted) return;

    setState(() {
      loading = false;
      hasError = failed;
      result = message;
    });
  }

  @override
  void dispose() {
    for (final controller in controllers.values) {
      controller.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Business Sales Estimate'),
        backgroundColor: const Color(0xFFE9EFEC),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                'Enter your business details',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                'Based on historical Kenyan business data. '
                'Estimates are for reference and are not guaranteed sales.',
              ),
              const SizedBox(height: 24),
              for (final name in controllers.keys)
                Padding(
                  padding: const EdgeInsets.only(bottom: 18),
                  child: TextField(
                    controller: controllers[name],
                    enabled: !loading,
                    keyboardType: [
                      'owner_age',
                      'employees',
                      'hours_worked_last_week',
                    ].contains(name)
                        ? const TextInputType.numberWithOptions(
                            decimal: true,
                            signed: true,
                          )
                        : TextInputType.text,
                    decoration: InputDecoration(
                      labelText: labels[name],
                      border: const OutlineInputBorder(),
                    ),
                  ),
                ),
              FilledButton(
                onPressed: loading ? null : predict,
                child: const Text('Predict'),
              ),
              const SizedBox(height: 16),
              if (loading) const LinearProgressIndicator(),
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: hasError
                      ? const Color(0xFFF8ECEC)
                      : const Color(0xFFEDF1EF),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  result,
                  style: TextStyle(
                    fontSize: 16,
                    color: hasError
                        ? const Color(0xFF923535)
                        : const Color(0xFF293D35),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}