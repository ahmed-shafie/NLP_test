import 'dart:convert';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import '../models/conversation.dart';
import '../services/api_client.dart';
import '../services/settings_service.dart';
import 'settings_screen.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key, required this.settings});

  final SettingsService settings;

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  late final ApiClient _api = ApiClient(widget.settings);
  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();
  final TextEditingController _textCtrl = TextEditingController();
  final ScrollController _scrollCtrl = ScrollController();

  final List<ChatMessage> _messages = [];
  String? _sessionId;
  ConversationSlots _slots = const ConversationSlots();
  String _status = 'collecting';
  bool _busy = false;
  bool _recording = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _messages.add(ChatMessage(
      sender: Sender.assistant,
      text:
          'Hi! I can set up a money transfer. Tap the mic and tell me the amount, '
          'currency, and who to send it to — or type below.',
    ));
  }

  @override
  void dispose() {
    _recorder.dispose();
    _player.dispose();
    _textCtrl.dispose();
    _scrollCtrl.dispose();
    super.dispose();
  }

  void _scrollToEnd() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.animateTo(
          _scrollCtrl.position.maxScrollExtent,
          duration: const Duration(milliseconds: 250),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _applyResponse(ConversationResponse r) {
    setState(() {
      _sessionId = r.sessionId;
      _slots = r.slots;
      _status = r.status;
      if (r.transcript != null && r.transcript!.isNotEmpty) {
        _messages.add(ChatMessage(
          sender: Sender.user,
          text: r.transcript!,
          isArabic: r.isArabic,
        ));
      }
      _messages.add(ChatMessage(
        sender: Sender.assistant,
        text: r.reply,
        isArabic: r.isArabic,
      ));
    });
    _scrollToEnd();
    _playReply(r);
  }

  Future<void> _playReply(ConversationResponse r) async {
    final b64 = r.audioBase64;
    if (b64 == null || b64.isEmpty) return;
    try {
      final bytes = base64Decode(b64);
      await _player.play(
        BytesSource(bytes, mimeType: r.audioMime ?? 'audio/mpeg'),
      );
    } catch (_) {
      // Playback is best-effort; the text reply is already shown.
    }
  }

  Future<void> _sendText() async {
    final text = _textCtrl.text.trim();
    if (text.isEmpty || _busy) return;
    _textCtrl.clear();
    setState(() {
      _messages.add(ChatMessage(sender: Sender.user, text: text));
      _busy = true;
      _error = null;
    });
    _scrollToEnd();
    try {
      _applyResponse(await _api.sendText(text, _sessionId));
    } on ApiException catch (e) {
      setState(() => _error = e.message);
    } catch (e) {
      setState(() => _error = 'Network error: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _toggleRecording() async {
    if (_busy) return;
    if (_recording) {
      await _stopAndSend();
      return;
    }
    if (!await _recorder.hasPermission()) {
      setState(() => _error = 'Microphone permission denied.');
      return;
    }
    final dir = await getTemporaryDirectory();
    final path =
        '${dir.path}/turn_${DateTime.now().millisecondsSinceEpoch}.m4a';
    await _recorder.start(
      const RecordConfig(encoder: AudioEncoder.aacLc),
      path: path,
    );
    setState(() {
      _recording = true;
      _error = null;
    });
  }

  Future<void> _stopAndSend() async {
    final path = await _recorder.stop();
    setState(() {
      _recording = false;
      if (path != null) _busy = true;
    });
    if (path == null) return;
    try {
      _applyResponse(await _api.sendVoice(path, _sessionId));
    } on ApiException catch (e) {
      setState(() => _error = e.message);
    } catch (e) {
      setState(() => _error = 'Network error: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _newConversation() {
    setState(() {
      _sessionId = null;
      _slots = const ConversationSlots();
      _status = 'collecting';
      _error = null;
      _messages
        ..clear()
        ..add(ChatMessage(
          sender: Sender.assistant,
          text: 'New conversation started. Tell me about your transfer.',
        ));
    });
  }

  Future<void> _openSettings() async {
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => SettingsScreen(settings: widget.settings),
      ),
    );
    setState(() {}); // reflect any base-url change
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Banking NLU Assistant'),
        actions: [
          IconButton(
            tooltip: 'New conversation',
            icon: const Icon(Icons.refresh),
            onPressed: _newConversation,
          ),
          IconButton(
            tooltip: 'Settings',
            icon: const Icon(Icons.settings),
            onPressed: _openSettings,
          ),
        ],
      ),
      body: Column(
        children: [
          _StatusBar(status: _status, slots: _slots),
          if (_error != null)
            Container(
              width: double.infinity,
              color: Theme.of(context).colorScheme.errorContainer,
              padding: const EdgeInsets.all(10),
              child: Text(
                _error!,
                style: TextStyle(
                  color: Theme.of(context).colorScheme.onErrorContainer,
                ),
              ),
            ),
          Expanded(
            child: ListView.builder(
              controller: _scrollCtrl,
              padding: const EdgeInsets.all(12),
              itemCount: _messages.length,
              itemBuilder: (_, i) => _Bubble(message: _messages[i]),
            ),
          ),
          if (_busy) const LinearProgressIndicator(minHeight: 2),
          _Composer(
            controller: _textCtrl,
            recording: _recording,
            busy: _busy,
            onSend: _sendText,
            onMic: _toggleRecording,
          ),
        ],
      ),
    );
  }
}

class _StatusBar extends StatelessWidget {
  const _StatusBar({required this.status, required this.slots});

  final String status;
  final ConversationSlots slots;

  Color _statusColor(BuildContext context) {
    switch (status) {
      case 'confirming':
        return Colors.amber.shade700;
      case 'completed':
        return Colors.green.shade600;
      case 'cancelled':
        return Colors.grey;
      default:
        return Theme.of(context).colorScheme.primary;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      color: Theme.of(context).colorScheme.surfaceContainerHighest,
      child: Row(
        children: [
          Chip(
            label: Text(status, style: const TextStyle(color: Colors.white)),
            backgroundColor: _statusColor(context),
            visualDensity: VisualDensity.compact,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Wrap(
              spacing: 6,
              runSpacing: 4,
              children: [
                _SlotChip(label: 'amount', value: slots.amount),
                _SlotChip(label: 'currency', value: slots.currency),
                _SlotChip(label: 'recipient', value: slots.recipient),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _SlotChip extends StatelessWidget {
  const _SlotChip({required this.label, required this.value});

  final String label;
  final String? value;

  @override
  Widget build(BuildContext context) {
    final filled = value != null && value!.isNotEmpty;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: filled ? Colors.green : Theme.of(context).dividerColor,
        ),
      ),
      child: Text(
        '$label: ${filled ? value : '—'}',
        style: const TextStyle(fontSize: 12),
      ),
    );
  }
}

class _Bubble extends StatelessWidget {
  const _Bubble({required this.message});

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final isUser = message.sender == Sender.user;
    final scheme = Theme.of(context).colorScheme;
    final bubble = Container(
      margin: const EdgeInsets.symmetric(vertical: 4),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      constraints: BoxConstraints(
        maxWidth: MediaQuery.of(context).size.width * 0.78,
      ),
      decoration: BoxDecoration(
        color: isUser ? scheme.primary : scheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Text(
        message.text,
        textDirection: message.isArabic ? TextDirection.rtl : null,
        style: TextStyle(
          color: isUser ? scheme.onPrimary : scheme.onSurface,
        ),
      ),
    );
    return Row(
      mainAxisAlignment:
          isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
      children: [bubble],
    );
  }
}

class _Composer extends StatelessWidget {
  const _Composer({
    required this.controller,
    required this.recording,
    required this.busy,
    required this.onSend,
    required this.onMic,
  });

  final TextEditingController controller;
  final bool recording;
  final bool busy;
  final VoidCallback onSend;
  final VoidCallback onMic;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(8, 6, 8, 8),
        child: Row(
          children: [
            IconButton.filled(
              onPressed: busy ? null : onMic,
              isSelected: recording,
              style: IconButton.styleFrom(
                backgroundColor: recording
                    ? Theme.of(context).colorScheme.error
                    : Theme.of(context).colorScheme.primary,
                foregroundColor: Colors.white,
              ),
              icon: Icon(recording ? Icons.stop : Icons.mic),
              tooltip: recording ? 'Stop & send' : 'Hold a conversation by voice',
            ),
            const SizedBox(width: 6),
            Expanded(
              child: TextField(
                controller: controller,
                textInputAction: TextInputAction.send,
                onSubmitted: (_) => onSend(),
                decoration: InputDecoration(
                  hintText: recording
                      ? 'Listening… tap stop to send'
                      : 'Type a message…',
                  border: const OutlineInputBorder(),
                  contentPadding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                ),
              ),
            ),
            const SizedBox(width: 6),
            IconButton(
              onPressed: busy ? null : onSend,
              icon: const Icon(Icons.send),
              tooltip: 'Send',
            ),
          ],
        ),
      ),
    );
  }
}
