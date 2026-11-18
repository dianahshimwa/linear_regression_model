import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

class ApiService {
  static const baseUrl =
      'https://women-enterprise-sales.onrender.com';

  static Future<double> predict(Map<String, dynamic> inputs) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(inputs),
      ).timeout(const Duration(seconds: 90));

      dynamic body;

      try {
        body = jsonDecode(response.body);
      } on FormatException {
        throw const FormatException(
          'Unexpected server response. Please try again shortly.',
        );
      }

      if (response.statusCode != 200) {
        throw FormatException(readError(body));
      }

      final prediction = body is Map
          ? body['predicted_weekly_sales_ksh']
          : null;

      if (prediction is! num || !prediction.isFinite) {
        throw const FormatException(
          'The server returned an invalid estimate.',
        );
      }

      return prediction.toDouble();
    } on TimeoutException {
      throw const FormatException(
        'The request timed out. Wait a moment and try again.',
      );
    } on http.ClientException {
      throw const FormatException(
        'Could not connect. Check your internet and try again.',
      );
    }
  }

  static String readError(dynamic body) {
    if (body is Map) {
      final detail = body['detail'];

      if (detail is String) return detail;

      if (detail is List) {
        return detail.map((error) {
          if (error is Map) {
            final location = error['loc'];
            final field = location is List && location.isNotEmpty
                ? location.last
                : 'Input';

            return '$field: ${error['msg']}';
          }

          return 'An input was not accepted.';
        }).join('\n');
      }
    }

    return 'The server could not process your request. Please try again.';
  }
}