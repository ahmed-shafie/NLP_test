// Data models mirroring the backend conversation/voice responses.

/// The slot values collected so far for a transfer.
class ConversationSlots {
  const ConversationSlots({
    this.amount,
    this.currency,
    this.recipient,
    this.sourceAccount,
    this.accountNumber,
    this.note,
  });

  final String? amount;
  final String? currency;
  final String? recipient;
  final String? sourceAccount;
  final String? accountNumber;
  final String? note;

  factory ConversationSlots.fromJson(Map<String, dynamic>? json) {
    if (json == null) return const ConversationSlots();
    return ConversationSlots(
      amount: json['amount'] as String?,
      currency: json['currency'] as String?,
      recipient: json['recipient'] as String?,
      sourceAccount: json['source_account'] as String?,
      accountNumber: json['account_number'] as String?,
      note: json['note'] as String?,
    );
  }
}

/// The response returned by `/conversation/text` and `/conversation/voice`.
class ConversationResponse {
  const ConversationResponse({
    required this.sessionId,
    required this.reply,
    required this.status,
    required this.language,
    required this.slots,
    this.intent,
    this.pendingSlot,
    this.complete = false,
    this.transcript,
    this.audioBase64,
    this.audioMime,
  });

  final String sessionId;
  final String reply;
  final String status;
  final String language;
  final ConversationSlots slots;
  final String? intent;
  final String? pendingSlot;
  final bool complete;

  /// Voice-only: what the speech recogniser heard.
  final String? transcript;

  /// Voice-only: base64-encoded synthesized reply audio.
  final String? audioBase64;
  final String? audioMime;

  bool get isArabic => language == 'ar';

  factory ConversationResponse.fromJson(Map<String, dynamic> json) {
    return ConversationResponse(
      sessionId: json['session_id'] as String? ?? '',
      reply: json['reply'] as String? ?? '',
      status: json['status'] as String? ?? 'collecting',
      language: json['language'] as String? ?? 'en',
      slots: ConversationSlots.fromJson(
        json['slots'] as Map<String, dynamic>?,
      ),
      intent: json['intent'] as String?,
      pendingSlot: json['pending_slot'] as String?,
      complete: json['complete'] as bool? ?? false,
      transcript: json['transcript'] as String?,
      audioBase64: json['audio_base64'] as String?,
      audioMime: json['audio_mime'] as String?,
    );
  }
}

/// Who authored a chat message.
enum Sender { user, assistant }

/// A single rendered chat bubble.
class ChatMessage {
  ChatMessage({
    required this.sender,
    required this.text,
    this.isArabic = false,
  });

  final Sender sender;
  final String text;
  final bool isArabic;
}
