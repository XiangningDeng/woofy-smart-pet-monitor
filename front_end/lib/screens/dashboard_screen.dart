import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});
  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  List<Map<String, dynamic>> activities = [];

  Future<void> fetchActivities() async {
    final response = await http.get(Uri.parse("http://3.133.137.212:5000/activities"));
    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body);
      setState(() {
        activities = data.cast<Map<String, dynamic>>();
      });
    } else {
      print("Failed to load activities");
    }
  }

  @override
  void initState() {
    super.initState();
    fetchActivities();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Woofy Activity Monitor")),
      body: RefreshIndicator(
        onRefresh: fetchActivities,
        child: ListView.builder(
          itemCount: activities.length,
          itemBuilder: (context, index) {
            final activity = activities[index];
            return ListTile(
              title: Text(activity['activity']),
              subtitle: Text("Device: ${activity['device_id']} • ${activity['timestamp']}"),
              leading: const Icon(Icons.pets),
            );
          },
        ),
      ),
    );
  }
}
