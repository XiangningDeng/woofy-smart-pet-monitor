import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:http/http.dart' as http;
import 'package:collection/collection.dart';
import 'package:provider/provider.dart';
import 'package:workmanager/workmanager.dart';
import 'config/language_config.dart';
import 'config/theme_config.dart';
import 'config/activity_translations.dart';
import 'services/settings_service.dart';

void callbackDispatcher() {
  Workmanager().executeTask((taskName, inputData) async {
    try {
      final response = await http.get(Uri.parse("https://woofy.it.com/api/all"));
      if (response.statusCode == 200) {
        print('Background fetch successful');
        return true;
      }
    } catch (e) {
      print('Background fetch failed: $e');
    }
    return false;
  });
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // 初始化 workmanager
  await Workmanager().initialize(
    callbackDispatcher,
    isInDebugMode: false,
  );
  
  // 只在 Android 上注册后台任务
  if (defaultTargetPlatform == TargetPlatform.android) {
    await Workmanager().registerPeriodicTask(
      "1",
      "fetchData",
      frequency: const Duration(minutes: 15),
      constraints: Constraints(
        networkType: NetworkType.connected,
        requiresBatteryNotLow: true,
      ),
    );
  }

  runApp(
    ChangeNotifierProvider(
      create: (_) => SettingsService(),
      child: const MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    final settings = Provider.of<SettingsService>(context);
    
    // 根据当前主题设置状态栏样式
    SystemChrome.setSystemUIOverlayStyle(
      settings.isDarkMode 
        ? SystemUiOverlayStyle.dark.copyWith(
            statusBarColor: Colors.transparent,
            statusBarBrightness: Brightness.light,
            statusBarIconBrightness: Brightness.light,
          )
        : SystemUiOverlayStyle.light.copyWith(
            statusBarColor: Colors.transparent,
            statusBarBrightness: Brightness.dark,
            statusBarIconBrightness: Brightness.dark,
          ),
    );

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeConfig.getLightTheme(),
      darkTheme: ThemeConfig.getDarkTheme(),
      themeMode: settings.isDarkMode ? ThemeMode.dark : ThemeMode.light,
      home: const DeviceSelectionPage(),
    );
  }
}

class DeviceSelectionPage extends StatefulWidget {
  const DeviceSelectionPage({super.key});

  @override
  State<DeviceSelectionPage> createState() => _DeviceSelectionPageState();
}

class _DeviceSelectionPageState extends State<DeviceSelectionPage> with WidgetsBindingObserver {
  final String apiUrl = "https://woofy.it.com/api/all";
  List<Map<String, dynamic>> allData = [];
  String? selectedDevice;
  bool isLoading = false;
  String? error;
  Timer? _refreshTimer;
  bool _isInBackground = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _startDataRefresh();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _refreshTimer?.cancel();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);
    setState(() {
      _isInBackground = state == AppLifecycleState.paused || 
                        state == AppLifecycleState.inactive;
    });

    if (state == AppLifecycleState.resumed) {
      // 应用从后台恢复时立即刷新数据
      fetchData();
    }
  }

  void _startDataRefresh() {
    fetchData();
    _refreshTimer = Timer.periodic(const Duration(seconds: 30), (_) {
      if (!_isInBackground) {
        print('Refreshing device list...');
        fetchData();
      }
    });
  }

  Future<void> fetchData() async {
    if (isLoading) return;

    try {
      setState(() {
        isLoading = true;
      });
      
      print('Fetching device data from API...');
      final res = await http.get(Uri.parse(apiUrl));
      
      if (!mounted) return;

      if (res.statusCode == 200) {
        List<dynamic> jsonList = json.decode(res.body);
        final newData = List<Map<String, dynamic>>.from(jsonList);
        if (mounted) {
          setState(() {
            allData = newData;
            error = null;
            isLoading = false;
          });
          print('Device list updated successfully');
        }
      } else {
        print('Failed to fetch device data: ${res.statusCode}');
        if (mounted) {
          setState(() {
            error = LanguageConfig.getTranslation('errorLoading', context.read<SettingsService>().currentLanguage);
            isLoading = false;
          });
        }
      }
    } catch (e) {
      print('Error fetching device data: $e');
      if (mounted) {
        setState(() {
          error = LanguageConfig.getTranslation('errorLoading', context.read<SettingsService>().currentLanguage);
          isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final settings = Provider.of<SettingsService>(context);
    final language = settings.currentLanguage;
    final List<String> deviceIds = allData.map((e) => e["device_id"].toString()).toSet().toList();

    return Scaffold(
      appBar: AppBar(
        title: Text(
          LanguageConfig.getTranslation('selectDevice', language),
          style: TextStyle(
            color: settings.isDarkMode ? Colors.white : Colors.blue[600],
            fontWeight: FontWeight.bold,
          ),
        ),
        backgroundColor: settings.isDarkMode ? const Color(0xFF1A1A1A) : Colors.white,
        elevation: 0,
        systemOverlayStyle: settings.isDarkMode
          ? SystemUiOverlayStyle.light.copyWith(
              statusBarColor: Colors.transparent,
              systemNavigationBarColor: Colors.black,
              statusBarBrightness: Brightness.dark,
              statusBarIconBrightness: Brightness.light,
            )
          : SystemUiOverlayStyle.dark.copyWith(
              statusBarColor: Colors.transparent,
              systemNavigationBarColor: Colors.white,
              statusBarBrightness: Brightness.light,
              statusBarIconBrightness: Brightness.dark,
            ),
        actions: [
          IconButton(
            icon: Icon(
              Icons.language,
              color: settings.isDarkMode ? Colors.white : Colors.blue[600],
            ),
            onPressed: () {
              showDialog(
                context: context,
                builder: (BuildContext context) {
                  return AlertDialog(
                    backgroundColor: settings.isDarkMode ? const Color(0xFF1A1A1A) : Colors.white,
                    title: Text(
                      LanguageConfig.getTranslation('language', language),
                      style: TextStyle(
                        color: settings.isDarkMode ? Colors.white : Colors.blue[600],
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    content: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        ListTile(
                          title: Text(
                            'English',
                            style: TextStyle(
                              color: settings.isDarkMode ? Colors.white : Colors.black87,
                            ),
                          ),
                          trailing: language == 'en' ? Icon(
                            Icons.check,
                            color: settings.isDarkMode ? Colors.white : Colors.blue[600],
                          ) : null,
                          onTap: () {
                            settings.setLanguage('en');
                            Navigator.pop(context);
                          },
                        ),
                        ListTile(
                          title: Text(
                            '中文',
                            style: TextStyle(
                              color: settings.isDarkMode ? Colors.white : Colors.black87,
                            ),
                          ),
                          trailing: language == 'zh' ? Icon(
                            Icons.check,
                            color: settings.isDarkMode ? Colors.white : Colors.blue[600],
                          ) : null,
                          onTap: () {
                            settings.setLanguage('zh');
                            Navigator.pop(context);
                          },
                        ),
                      ],
                    ),
                  );
                },
              );
            },
          ),
          IconButton(
            icon: Stack(
              children: [
                Icon(
                  Icons.refresh,
                  color: settings.isDarkMode ? Colors.white : Colors.blue[600],
                ),
                if (isLoading)
                  Positioned.fill(
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(
                        settings.isDarkMode ? Colors.white : Colors.blue[600]!,
                      ),
                    ),
                  ),
              ],
            ),
            onPressed: isLoading ? null : () => fetchData(),
          ),
        ],
      ),
      body: error != null
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(error!, style: const TextStyle(color: Colors.red)),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: () => fetchData(),
                    child: Text(LanguageConfig.getTranslation('retry', language)),
                  ),
                ],
              ),
            )
          : Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (isLoading && allData.isEmpty)
                    const Center(child: CircularProgressIndicator())
                  else if (deviceIds.isEmpty)
                    Center(child: Text(LanguageConfig.getTranslation('noDevices', language)))
                  else ...[
                    Text(
                      LanguageConfig.getTranslation('selectDeviceToMonitor', language),
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: settings.isDarkMode ? Colors.white : Colors.blue[600],
                      ),
                    ),
                    const SizedBox(height: 20),
                    DropdownButtonFormField<String>(
                      value: selectedDevice,
                      decoration: InputDecoration(
                        border: const OutlineInputBorder(),
                        labelText: LanguageConfig.getTranslation('selectDevice', language),
                        labelStyle: TextStyle(
                          color: settings.isDarkMode ? Colors.white70 : Colors.blue[600],
                        ),
                      ),
                      dropdownColor: settings.isDarkMode ? const Color(0xFF1A1A1A) : Colors.white,
                      style: TextStyle(
                        color: settings.isDarkMode ? Colors.white : Colors.blue[600],
                        fontSize: 16,
                      ),
                      items: deviceIds.map((id) {
                        return DropdownMenuItem(
                          value: id,
                          child: Text(
                            "${LanguageConfig.getTranslation('device', language)}: $id",
                            style: TextStyle(
                              color: settings.isDarkMode ? Colors.white : Colors.blue[600],
                            ),
                          ),
                        );
                      }).toList(),
                      onChanged: (value) {
                        setState(() {
                          selectedDevice = value;
                        });
                      },
                    ),
                    const SizedBox(height: 20),
                    ElevatedButton(
                      onPressed: selectedDevice == null
                          ? null
                          : () {
                              Navigator.pushReplacement(
                                context,
                                MaterialPageRoute(
                                  builder: (context) => MainScreen(
                                    initialDevice: selectedDevice!,
                                    allData: allData,
                                  ),
                                ),
                              );
                            },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: settings.isDarkMode ? Colors.blue : Colors.blue[600],
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                      ),
                      child: Text(LanguageConfig.getTranslation('continue', language)),
                    ),
                  ],
                ],
              ),
            ),
    );
  }
}

class MainScreen extends StatefulWidget {
  final String initialDevice;
  final List<Map<String, dynamic>> allData;

  const MainScreen({
    super.key,
    required this.initialDevice,
    required this.allData,
  });

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _selectedIndex = 0;
  Timer? _timer;
  List<Map<String, dynamic>> allData = [];
  String selectedDevice = "";
  bool isLoading = false;
  String? error;
  bool isDarkMode = false;
  String currentLanguage = 'en';
  
  // 定义固定的颜色映射
  final Map<String, Color> activityColors = {
    'Feeding': Colors.orange,
    'Playing': Colors.red,
    'Running': Colors.green,
    'Sniffing': Colors.purple,
    'Stilling': Colors.blue,
    'Walking': Colors.teal,
  };

  @override
  void initState() {
    super.initState();
    allData = widget.allData;
    selectedDevice = widget.initialDevice;
    _startDataRefresh();
  }

  Future<void> fetchData() async {
    if (isLoading) return;

    try {
      setState(() {
        isLoading = true;
      });
      
      print('Fetching data from API...');
      final res = await http.get(Uri.parse("https://woofy.it.com/api/all"));
      
      if (!mounted) return;
      
      if (res.statusCode == 200) {
        List<dynamic> jsonList = json.decode(res.body);
        final newData = List<Map<String, dynamic>>.from(jsonList);
        setState(() {
          allData = newData;
          error = null;
          isLoading = false;
        });
        print('Data refreshed successfully');
      } else {
        print('Failed to fetch data: ${res.statusCode}');
        setState(() {
          error = 'Failed to load data: ${res.statusCode}';
          isLoading = false;
        });
      }
    } catch (e) {
      print('Error fetching data: $e');
      if (mounted) {
        setState(() {
          error = 'Error fetching data: $e';
          isLoading = false;
        });
      }
    }
  }

  void _startDataRefresh() {
    fetchData();
    _timer = Timer.periodic(const Duration(seconds: 30), (_) {
      print('Refreshing data...');
      fetchData();
    });
  }

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final settings = Provider.of<SettingsService>(context);
    // 过滤当前设备的数据
    final List<Map<String, dynamic>> deviceData = allData
        .where((e) => e["device_id"].toString() == selectedDevice)
        .toList();
    print('Current device data length: ${deviceData.length}');

    final activityCount = <String, int>{};
    for (var row in deviceData) {
      String act = row["activity"];
      activityCount[act] = (activityCount[act] ?? 0) + 1;
    }

    final activityList = activityCount.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    final List<Widget> pages = [
      ActivityDurationPage(
        activityList: activityList,
        deviceData: deviceData,
        activityColors: activityColors,
        isDarkMode: settings.isDarkMode,
        currentLanguage: settings.currentLanguage,
      ),
      ActivityRecordsPage(
        deviceData: deviceData,
        isDarkMode: settings.isDarkMode,
        currentLanguage: settings.currentLanguage,
      ),
      SettingsPage(
        isDarkMode: settings.isDarkMode,
        currentLanguage: settings.currentLanguage,
        onThemeChanged: (value) {
          setState(() {
            settings.toggleTheme();
          });
        },
        onLanguageChanged: (value) {
          setState(() {
            settings.setLanguage(value);
          });
        },
      ),
    ];

    return WillPopScope(
      onWillPop: () async {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const DeviceSelectionPage()),
        );
        return false;
      },
      child: Scaffold(
        appBar: AppBar(
          title: Row(
            children: [
              Text(
                "${LanguageConfig.getTranslation('device', settings.currentLanguage)}: $selectedDevice",
                style: TextStyle(
                  color: settings.isDarkMode ? Colors.white : Colors.blue[600],
                  fontWeight: FontWeight.bold,
                ),
              ),
              if (isLoading)
                Padding(
                  padding: const EdgeInsets.only(left: 8.0),
                  child: SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(
                        settings.isDarkMode ? Colors.white : Colors.blue[600]!,
                      ),
                    ),
                  ),
                ),
            ],
          ),
          backgroundColor: settings.isDarkMode ? const Color(0xFF1A1A1A) : Colors.white,
          elevation: 0,
          systemOverlayStyle: settings.isDarkMode
            ? SystemUiOverlayStyle.light.copyWith(
                statusBarColor: Colors.transparent,
                systemNavigationBarColor: Colors.black,
                statusBarBrightness: Brightness.dark,
                statusBarIconBrightness: Brightness.light,
              )
            : SystemUiOverlayStyle.dark.copyWith(
                statusBarColor: Colors.transparent,
                systemNavigationBarColor: Colors.white,
                statusBarBrightness: Brightness.light,
                statusBarIconBrightness: Brightness.dark,
              ),
          leading: IconButton(
            icon: Icon(
              Icons.arrow_back,
              color: settings.isDarkMode ? Colors.white : Colors.blue[600],
            ),
            onPressed: () {
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(builder: (context) => const DeviceSelectionPage()),
              );
            },
          ),
          actions: [
            IconButton(
              icon: Stack(
                children: [
                  Icon(
                    Icons.refresh,
                    color: settings.isDarkMode ? Colors.white : Colors.blue[600],
                  ),
                  if (isLoading)
                    Positioned.fill(
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        valueColor: AlwaysStoppedAnimation<Color>(
                          settings.isDarkMode ? Colors.white : Colors.blue[600]!,
                        ),
                      ),
                    ),
                ],
              ),
              onPressed: isLoading ? null : () => fetchData(),
            ),
          ],
        ),
        body: error != null
            ? Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      error!,
                      style: TextStyle(
                        color: settings.isDarkMode ? Colors.red[300] : Colors.red,
                      ),
                    ),
                    const SizedBox(height: 16),
                    ElevatedButton(
                      onPressed: fetchData,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: settings.isDarkMode ? Colors.blue : Colors.blue[600],
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                      ),
                      child: Text(LanguageConfig.getTranslation('retry', settings.currentLanguage)),
                    ),
                  ],
                ),
              )
            : deviceData.isEmpty
                ? Center(
                    child: Text(
                      LanguageConfig.getTranslation('noData', settings.currentLanguage),
                      style: TextStyle(
                        color: settings.isDarkMode ? Colors.white70 : Colors.grey[600],
                        fontSize: 16,
                      ),
                    ),
                  )
                : pages[_selectedIndex],
        bottomNavigationBar: BottomNavigationBar(
          items: [
            BottomNavigationBarItem(
              icon: const Icon(Icons.pie_chart),
              label: settings.currentLanguage == 'en' ? 'Overview' : '概览',
            ),
            BottomNavigationBarItem(
              icon: const Icon(Icons.list),
              label: settings.currentLanguage == 'en' ? 'Records' : '记录',
            ),
            BottomNavigationBarItem(
              icon: const Icon(Icons.settings),
              label: settings.currentLanguage == 'en' ? 'Settings' : '设置',
            ),
          ],
          currentIndex: _selectedIndex,
          selectedItemColor: settings.isDarkMode ? Colors.white : Colors.blue[600],
          unselectedItemColor: settings.isDarkMode ? Colors.white60 : Colors.grey[600],
          backgroundColor: settings.isDarkMode ? const Color(0xFF1A1A1A) : Colors.white,
          onTap: _onItemTapped,
        ),
      ),
    );
  }
}

class ActivityDurationPage extends StatefulWidget {
  final List<MapEntry<String, int>> activityList;
  final List<Map<String, dynamic>> deviceData;
  final Map<String, Color> activityColors;
  final bool isDarkMode;
  final String currentLanguage;

  const ActivityDurationPage({
    super.key,
    required this.activityList,
    required this.deviceData,
    required this.activityColors,
    required this.isDarkMode,
    required this.currentLanguage,
  });

  @override
  State<ActivityDurationPage> createState() => _ActivityDurationPageState();
}

class _ActivityDurationPageState extends State<ActivityDurationPage> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  DateTime selectedDate = DateTime.now();
  List<Map<String, dynamic>> filteredData = [];
  List<Map<String, dynamic>> _previousData = [];
  List<ActivityBlock> _activityBlocks = [];
  int? _touchedIndex;
  String? _selectedActivity;
  bool _isTimelineExpanded = false;
  bool _isPieChartExpanded = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _filterDataByDate();
  }

  @override
  void didUpdateWidget(ActivityDurationPage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!const DeepCollectionEquality().equals(oldWidget.deviceData, widget.deviceData)) {
      _filterDataByDate();
    }
  }

  void _filterDataByDate() {
    final startOfDay = DateTime(selectedDate.year, selectedDate.month, selectedDate.day);
    final endOfDay = startOfDay.add(const Duration(days: 1));

    setState(() {
      filteredData = widget.deviceData.where((data) {
        final dataTime = DateTime.parse(data["timestamp"]);
        return dataTime.isAfter(startOfDay) && dataTime.isBefore(endOfDay);
      }).toList();
      _processActivityBlocks();
    });
  }

  void _processActivityBlocks() {
    if (filteredData.isEmpty) {
      _activityBlocks = [];
      return;
    }

    // 按时间排序
    filteredData.sort((a, b) => a["timestamp"].compareTo(b["timestamp"]));
    
    List<ActivityBlock> blocks = [];
    ActivityBlock? currentBlock;
    DateTime? lastTime;
    String? lastActivity;
    
    for (var data in filteredData) {
      final time = DateTime.parse(data["timestamp"]);
      final activity = data["activity"];
      
      // 如果是第一条数据
      if (lastTime == null) {
        currentBlock = ActivityBlock(
          activity: activity,
          startTime: time,
          endTime: time,
        );
        lastTime = time;
        lastActivity = activity;
        continue;
      }

      // 计算与上一条数据的时间差
      final duration = time.difference(lastTime);
      
      // 如果时间差超过1分钟，开始新的活动块
      if (duration.inSeconds > 60) {
        if (currentBlock != null) {
          currentBlock.endTime = lastTime!;
          blocks.add(currentBlock);
        }
        
        // 开始新的活动块
        currentBlock = ActivityBlock(
          activity: activity,
          startTime: time,
          endTime: time,
        );
      }
      // 如果活动发生变化
      else if (activity != lastActivity) {
        if (currentBlock != null) {
          currentBlock.endTime = time;
          blocks.add(currentBlock);
        }
        currentBlock = ActivityBlock(
          activity: activity,
          startTime: time,
          endTime: time,
        );
      } else {
        // 更新当前活动块的结束时间
        if (currentBlock != null) {
          currentBlock.endTime = time;
        }
      }
      
      lastTime = time;
      lastActivity = activity;
    }
    
    // 添加最后一个块
    if (currentBlock != null && lastTime != null) {
      currentBlock.endTime = lastTime;
      blocks.add(currentBlock);
    }
    
    setState(() {
      _activityBlocks = blocks;
    });
  }

  String _formatDuration(Duration duration) {
    final hours = duration.inHours;
    final minutes = duration.inMinutes.remainder(60);
    final seconds = duration.inSeconds.remainder(60);
    
    if (hours > 0) {
      return minutes > 0 ? '${hours}h ${minutes}m' : '${hours}h';
    } else if (minutes > 0) {
      return seconds > 0 ? '${minutes}m ${seconds}s' : '${minutes}m';
    } else {
      return '${seconds}s';
    }
  }

  List<FlSpot> _generateTimelineSpots(String activity) {
    if (_activityBlocks.isEmpty) return [];

    // 获取当天的起始时间作为参考点
    final firstTime = _activityBlocks.first.startTime;
    final dayStart = DateTime(firstTime.year, firstTime.month, firstTime.day);
    
    List<FlSpot> spots = [];
    double? lastX;
    
    // 添加初始点
    spots.add(FlSpot(0, 0));
    
    // 遍历每个时间块
    for (var block in _activityBlocks) {
      final startHours = block.startTime.difference(dayStart).inMinutes / 60.0;
      final endHours = block.endTime.difference(dayStart).inMinutes / 60.0;
      
      // 如果是当前活动，添加这个时间段
      if (block.activity == activity) {
        // 如果不是第一个点，且与上一个点有间隔，确保在起点添加0值
        if (lastX != null && lastX < startHours) {
          spots.add(FlSpot(lastX, 0));
          spots.add(FlSpot(startHours, 0));
        }
        // 添加活动开始点
        spots.add(FlSpot(startHours, 1));
        // 添加活动结束点
        spots.add(FlSpot(endHours, 1));
        spots.add(FlSpot(endHours, 0));
        lastX = endHours;
      }
    }
    
    // 添加最后一个点到24小时
    if (spots.isNotEmpty) {
      spots.add(FlSpot(24, 0));
    }
    
    return spots;
  }

  Widget _buildPieChart() {
    final totalDurations = _calculateTotalDurations();
    final totalSeconds = totalDurations.values.fold<int>(0, (sum, duration) => sum + duration.inSeconds);
    
    // 获取所有可能的活动类型（从activityColors中获取，确保顺序固定）
    final allActivities = widget.activityColors.keys.toList();
    
    // 过滤掉持续时间为0的活动，但保持顺序
    final validDurations = Map<String, Duration>.fromEntries(
      allActivities.map((activity) {
        final duration = totalDurations[activity] ?? Duration.zero;
        return MapEntry(activity, duration);
      }).where((entry) => entry.value.inSeconds > 0)
    );

    if (validDurations.isEmpty) return const SizedBox.shrink();

    return Column(
      children: [
        GestureDetector(
          onTap: () {
            setState(() {
              _isPieChartExpanded = !_isPieChartExpanded;
              if (!_isPieChartExpanded) {
                _selectedActivity = null;
              }
            });
          },
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            height: _isPieChartExpanded ? 240 : 180,
            padding: const EdgeInsets.symmetric(vertical: 16),
            child: PieChart(
              PieChartData(
                pieTouchData: PieTouchData(
                  touchCallback: (FlTouchEvent event, pieTouchResponse) {
                    setState(() {
                      if (!event.isInterestedForInteractions ||
                          pieTouchResponse == null ||
                          pieTouchResponse.touchedSection == null) {
                        _touchedIndex = null;
                        return;
                      }
                      _touchedIndex = pieTouchResponse.touchedSection!.touchedSectionIndex;
                      _selectedActivity = validDurations.keys.elementAt(_touchedIndex!);
                      _scrollToSelectedActivity();
                    });
                  },
                ),
                sectionsSpace: 3,
                centerSpaceRadius: _isPieChartExpanded ? 50 : 40,
                sections: validDurations.entries.map((entry) {
                  final isTouched = _touchedIndex == validDurations.keys.toList().indexOf(entry.key);
                  final isSelected = _selectedActivity == entry.key;
                  final radius = isSelected ? 85.0 : 70.0;
                  
                  return PieChartSectionData(
                    color: widget.activityColors[entry.key]?.withOpacity(
                      _selectedActivity == null || isSelected ? 1.0 : 0.3
                    ),
                    value: entry.value.inSeconds.toDouble(),
                    radius: isSelected ? radius * 1.01 : radius,
                    showTitle: false,
                    titleStyle: const TextStyle(fontSize: 0),
                  );
                }).toList(),
              ),
            ),
          ),
        ),
        const SizedBox(height: 32),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Wrap(
            spacing: 12,
            runSpacing: 12,
            children: allActivities.map((activity) {
              final duration = totalDurations[activity] ?? Duration.zero;
              final proportion = duration.inSeconds / (totalSeconds > 0 ? totalSeconds : 1);
              final isSelected = _selectedActivity == activity;
              
              if (duration.inSeconds == 0) {
                return Container(
                  height: 0,
                  width: 0,
                  margin: EdgeInsets.zero,
                  padding: EdgeInsets.zero,
                );
              }

              return AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                curve: Curves.easeInOut,
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                decoration: BoxDecoration(
                  color: widget.activityColors[activity]?.withOpacity(
                    _selectedActivity == null || isSelected ? 0.1 : 0.05
                  ),
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(
                    color: widget.activityColors[activity]?.withOpacity(
                      _selectedActivity == null || isSelected ? 1.0 : 0.3
                    ) ?? Colors.grey,
                    width: isSelected ? 2 : 1,
                  ),
                ),
                child: GestureDetector(
                  onTap: () {
                    setState(() {
                      _selectedActivity = activity;
                      _scrollToSelectedActivity();
                    });
                  },
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          color: widget.activityColors[activity],
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        "${ActivityTranslations.getTranslation(activity, widget.currentLanguage)} ${_formatDuration(duration)} (${(proportion * 100).round()}%)",
                        style: TextStyle(
                          fontSize: isSelected ? 14 : 12,
                          fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
                          color: widget.activityColors[activity] ?? Colors.grey,
                        ),
                      ),
                    ],
                  ),
                ),
              );
            }).toList(),
          ),
        ),
      ],
    );
  }

  void _scrollToSelectedActivity() {
    if (_selectedActivity != null) {
      final index = _activityBlocks.indexWhere((block) => block.activity == _selectedActivity);
      if (index != -1) {
        // 使用ScrollController滚动到指定位置
        _scrollController.animateTo(
          index * 60.0, // 假设每个项目高度为60
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeInOut,
        );
      }
    }
  }

  Widget _buildTimelineChart() {
    if (_activityBlocks.isEmpty) return const SizedBox.shrink();

    final firstTime = _activityBlocks.first.startTime;
    final lastTime = _activityBlocks.last.endTime;
    final dayStart = DateTime(firstTime.year, firstTime.month, firstTime.day);
    
    double startHour = firstTime.difference(dayStart).inMinutes / 60.0;
    double endHour = lastTime.difference(dayStart).inMinutes / 60.0;
    
    startHour = max(0, startHour - 0.5);
    endHour = min(24, endHour + 0.5);
    
    double timeRange = endHour - startHour;
    double interval;
    if (timeRange <= 2) {
      interval = 0.5;
    } else if (timeRange <= 4) {
      interval = 1.0;
    } else if (timeRange <= 8) {
      interval = 2.0;
    } else {
      interval = 3.0;
    }

    final validBlocks = _activityBlocks.where((block) {
      final duration = block.endTime.difference(block.startTime);
      return duration.inSeconds > 0;
    }).toList();

    return Container(
      height: 100,
      padding: const EdgeInsets.only(right: 16, top: 16, bottom: 16),
      child: Column(
        children: [
          Expanded(
            child: LayoutBuilder(
              builder: (context, constraints) {
                final double totalWidth = constraints.maxWidth;
                final double totalDuration = endHour - startHour;
                
                return Stack(
                  children: [
                    Positioned.fill(
                      child: GestureDetector(
                        onTapDown: (details) {
                          setState(() {
                            _selectedActivity = null;
                          });
                        },
                        child: Container(
                          decoration: BoxDecoration(
                            color: widget.isDarkMode ? Colors.grey[850] : Colors.grey[200],
                            borderRadius: BorderRadius.circular(4),
                          ),
                        ),
                      ),
                    ),
                    ...validBlocks.map((block) {
                      final blockStart = block.startTime.difference(dayStart).inMinutes / 60.0;
                      final blockEnd = block.endTime.difference(dayStart).inMinutes / 60.0;
                      final isSelected = _selectedActivity == block.activity;
                      
                      final left = (blockStart - startHour) / totalDuration * totalWidth;
                      final width = (blockEnd - blockStart) / totalDuration * totalWidth;
                      
                      return Positioned(
                        left: left,
                        top: 0,
                        bottom: 0,
                        width: width,
                        child: GestureDetector(
                          onTapDown: (details) {
                            setState(() {
                              _selectedActivity = block.activity;
                              _scrollToSelectedActivity();
                            });
                          },
                          child: AnimatedContainer(
                            duration: const Duration(milliseconds: 300),
                            transform: Matrix4.identity()..scale(1.0, isSelected ? 1.15 : 1.0, 1.0),
                            transformAlignment: Alignment.center,
                            decoration: BoxDecoration(
                              color: widget.activityColors[block.activity]?.withOpacity(
                                _selectedActivity == null || isSelected ? 1.0 : 0.3
                              ),
                            ),
                          ),
                        ),
                      );
                    }).toList(),
                  ],
                );
              },
            ),
          ),
          const SizedBox(height: 8),
          SizedBox(
            height: 20,
            child: LayoutBuilder(
              builder: (context, constraints) {
                final double totalWidth = constraints.maxWidth;
                final double totalDuration = endHour - startHour;
                final List<Widget> timeLabels = [];
                
                for (double time = startHour; time <= endHour; time += interval) {
                  final hours = time.toInt();
                  final minutes = ((time - hours) * 60).toInt();
                  final position = (time - startHour) / totalDuration * totalWidth;
                  
                  timeLabels.add(
                    Positioned(
                      left: position - 20,
                      child: Text(
                        '${hours.toString().padLeft(2, '0')}:${minutes.toString().padLeft(2, '0')}',
                        style: TextStyle(
                          fontSize: 10,
                          color: widget.isDarkMode ? Colors.grey[400] : Colors.grey[600],
                        ),
                      ),
                    ),
                  );
                }
                
                return Stack(children: timeLabels);
              },
            ),
          ),
        ],
      ),
    );
  }

  final ScrollController _scrollController = ScrollController();

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final totalDurations = _calculateTotalDurations();
    final totalSeconds = totalDurations.values.fold<int>(0, (sum, duration) => sum + duration.inSeconds);
    
    // 过滤掉持续时间为0的活动
    final validDurations = Map<String, Duration>.fromEntries(
      totalDurations.entries.where((entry) => entry.value.inSeconds > 0)
    );
    
    return GestureDetector(
      onTap: () {
        setState(() {
          _selectedActivity = null;
        });
      },
      behavior: HitTestBehavior.translucent,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                TextButton.icon(
                  icon: Icon(
                    Icons.calendar_today,
                    size: 20,
                    color: widget.isDarkMode ? Colors.white : Colors.blue[600],
                  ),
                  label: Text(
                    "${selectedDate.month.toString().padLeft(2, '0')}/${selectedDate.day.toString().padLeft(2, '0')}",
                    style: TextStyle(
                      fontSize: 14,
                      color: widget.isDarkMode ? Colors.white : Colors.blue[600],
                    ),
                  ),
                  style: TextButton.styleFrom(
                    backgroundColor: widget.isDarkMode ? Colors.grey[800] : Colors.blue[50],
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  onPressed: () async {
                    final DateTime? picked = await showDatePicker(
                      context: context,
                      initialDate: selectedDate,
                      firstDate: DateTime(2024),
                      lastDate: DateTime.now(),
                    );
                    if (picked != null && picked != selectedDate) {
                      setState(() {
                        selectedDate = picked;
                      });
                      _filterDataByDate();
                    }
                  },
                ),
              ],
            ),
            const SizedBox(height: 8),
            if (totalDurations.isNotEmpty) ...[
              _buildPieChart(),
              const SizedBox(height: 12),
              _buildTimelineChart(),
              const SizedBox(height: 12),
            ],
            Expanded(
              child: _activityBlocks.isEmpty
                ? Center(child: Text(LanguageConfig.getTranslation('noDataForDate', widget.currentLanguage)))
                : Stack(
                  children: [
                    Positioned.fill(
                      child: GestureDetector(
                        onTap: () {
                          setState(() {
                            _selectedActivity = null;
                          });
                        },
                      ),
                    ),
                    ListView.builder(
                      controller: _scrollController,
                      itemCount: _selectedActivity == null 
                          ? _activityBlocks.length 
                          : _activityBlocks.where((block) => block.activity == _selectedActivity).length,
                      itemBuilder: (context, index) {
                        final displayBlocks = _selectedActivity == null 
                            ? _activityBlocks 
                            : _activityBlocks.where((block) => block.activity == _selectedActivity).toList();
                        
                        if (displayBlocks.isEmpty || index >= displayBlocks.length) {
                          return const SizedBox.shrink();
                        }

                        final block = displayBlocks[index];
                        final duration = block.endTime.difference(block.startTime);
                        final isSelected = _selectedActivity == block.activity;
                        
                        return GestureDetector(
                          behavior: HitTestBehavior.opaque,
                          onTap: () {
                            setState(() {
                              _selectedActivity = block.activity;
                            });
                          },
                          child: Container(
                            height: 60,
                            padding: const EdgeInsets.symmetric(vertical: 4.0),
                            child: Row(
                              children: [
                                SizedBox(
                                  width: 100,
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Text(
                                        '${block.startTime.hour.toString().padLeft(2, '0')}:${block.startTime.minute.toString().padLeft(2, '0')}',
                                        style: const TextStyle(
                                          fontSize: 14,
                                          fontWeight: FontWeight.w500,
                                        ),
                                      ),
                                      Text(
                                        _formatDuration(duration),
                                        style: TextStyle(
                                          fontSize: 12,
                                          color: Colors.grey[600],
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                                Expanded(
                                  child: Container(
                                    height: 40,
                                    decoration: BoxDecoration(
                                      color: widget.activityColors[block.activity]?.withOpacity(
                                        _selectedActivity == null || isSelected ? 0.2 : 0.1
                                      ),
                                      borderRadius: BorderRadius.circular(4),
                                      border: Border.all(
                                        color: widget.activityColors[block.activity]?.withOpacity(
                                          _selectedActivity == null || isSelected ? 1.0 : 0.3
                                        ) ?? Colors.grey,
                                        width: isSelected ? 3 : 2,
                                      ),
                                    ),
                                    child: Center(
                                      child: Text(
                                        ActivityTranslations.getTranslation(block.activity, widget.currentLanguage),
                                        style: TextStyle(
                                          fontSize: isSelected ? 16 : 14,
                                          fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
                                          color: widget.activityColors[block.activity]?.withOpacity(
                                            _selectedActivity == null || isSelected ? 1.0 : 0.3
                                          ) ?? Colors.grey,
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ],
                ),
            ),
          ],
        ),
      ),
    );
  }

  Map<String, Duration> _calculateTotalDurations() {
    final Map<String, Duration> totals = {};
    for (var block in _activityBlocks) {
      final duration = block.endTime.difference(block.startTime);
      totals[block.activity] = (totals[block.activity] ?? Duration.zero) + duration;
    }
    return totals;
  }
}

class ActivityBlock {
  final String activity;
  final DateTime startTime;
  DateTime endTime;

  ActivityBlock({
    required this.activity,
    required this.startTime,
    required this.endTime,
  });
}

class ActivityRecordsPage extends StatelessWidget {
  final List<Map<String, dynamic>> deviceData;
  final bool isDarkMode;
  final String currentLanguage;

  const ActivityRecordsPage({
    super.key,
    required this.deviceData,
    required this.isDarkMode,
    required this.currentLanguage,
  });

  @override
  Widget build(BuildContext context) {
    final sortedData = List<Map<String, dynamic>>.from(deviceData)
      ..sort((a, b) => b["timestamp"].compareTo(a["timestamp"]));

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            currentLanguage == 'en' ? "🕒 Recent Activity Records" : "🕒 最近活动记录",
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)
          ),
          const SizedBox(height: 20),
          Expanded(
            child: sortedData.isEmpty
              ? Center(child: Text(currentLanguage == 'en' ? "No data available" : "暂无数据"))
              : ListView.builder(
                  itemCount: sortedData.length < 10 ? sortedData.length : 10,
                  itemBuilder: (context, index) {
                    final row = sortedData[index];
                    return Card(
                      margin: const EdgeInsets.only(bottom: 8),
                      child: ListTile(
                        leading: const Icon(Icons.pets, color: Colors.blueAccent),
                        title: Text(
                          row["activity"],
                          style: const TextStyle(fontWeight: FontWeight.bold),
                        ),
                        subtitle: Text(
                          _formatTimestamp(row["timestamp"]),
                          style: TextStyle(color: Colors.grey[600]),
                        ),
                      ),
                    );
                  },
                ),
          ),
        ],
      ),
    );
  }

  String _formatTimestamp(String timestamp) {
    try {
      final dateTime = DateTime.parse(timestamp);
      return '${dateTime.year}-${dateTime.month.toString().padLeft(2, '0')}-${dateTime.day.toString().padLeft(2, '0')} '
          '${dateTime.hour.toString().padLeft(2, '0')}:${dateTime.minute.toString().padLeft(2, '0')}:${dateTime.second.toString().padLeft(2, '0')}';
    } catch (e) {
      return timestamp;
    }
  }
}

class SettingsPage extends StatelessWidget {
  final bool isDarkMode;
  final String currentLanguage;
  final Function(bool) onThemeChanged;
  final Function(String) onLanguageChanged;

  const SettingsPage({
    super.key,
    required this.isDarkMode,
    required this.currentLanguage,
    required this.onThemeChanged,
    required this.onLanguageChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            currentLanguage == 'en' ? 'Appearance' : '外观',
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          SwitchListTile(
            title: Text(currentLanguage == 'en' ? 'Dark Mode' : '深色模式'),
            value: isDarkMode,
            onChanged: onThemeChanged,
          ),
          const Divider(),
          const SizedBox(height: 16),
          Text(
            currentLanguage == 'en' ? 'Language' : '语言',
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          ListTile(
            title: Text(currentLanguage == 'en' ? 'English' : '英语'),
            trailing: currentLanguage == 'en' ? const Icon(Icons.check) : null,
            onTap: () => onLanguageChanged('en'),
          ),
          ListTile(
            title: Text(currentLanguage == 'en' ? 'Chinese' : '中文'),
            trailing: currentLanguage == 'zh' ? const Icon(Icons.check) : null,
            onTap: () => onLanguageChanged('zh'),
          ),
        ],
      ),
    );
  }
}