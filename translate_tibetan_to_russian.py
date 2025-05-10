from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Проверяем, доступен ли GPU (CUDA) или MPS (для Apple Silicon), иначе используем CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Используемое устройство: {device}")

# Выберите модель NLLB.
model_name = 'facebook/nllb-200-3.3B'# Начнем с меньшей модели
# model_name = 'facebook/nllb-200-1.3B' # Модель побольше
# model_name = 'facebook/nllb-200-3.3B' # Самая большая из этих опций

print(f"Загрузка токенизатора для {model_name}...")
# Укажите язык источника (тибетский) при загрузке токенизатора
# Код для стандартного тибетского: bod_Tibt
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="bod_Tibt")

print(f"Загрузка модели {model_name}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval() # Перевод модели в режим оценки

# Тибетский философский текст для перевода:
tibetan_text = """དལ་འབྱོར་གྱི་མི་ལུས་རིན་པོ་ཆེ་འདི་ནི་ཐོབ་པར་ཤིན་ཏུ་དཀའ་བ་ཡིན། རྒྱུ་མཚན་ནི་འགྲོ་བ་རིགས་དྲུག་གི་ནང་ནས་མི་ལུས་ཐོབ་པ་ཉུང་ཞིང་། དེ་ལས་ཀྱང་ཆོས་བྱེད་པའི་དལ་ཁོམ་དང་འབྱོར་པ་ཚང་བ་ནི་དེ་བས་ཀྱང་དཀོན་པའི་ཕྱིར་རོ། །ལུས་འདི་ལན་ཅིག་ཐོབ་ཀྱང་རྟག་པ་མེད་པས་སྐད་ཅིག་ཀྱང་མི་སྡོད་པར་འཆི་བ་ལ་ཐུག་ཡོད། དེས་ན་ཚེ་འདི་ལ་དོན་མེད་པའི་བྱ་བ་ཁོ་ནས་དུས་འདའ་བར་མི་བྱ་བར། རང་གཞན་གཉིས་ཀྱི་དོན་ཆེན་པོ་སྒྲུབ་པའི་ཐབས་ལ་བརྩོན་པར་བྱའོ། །གལ་ཏེ་ད་ལྟ་ཆོས་ལ་མ་འབད་ན་ཕྱིས་ནམ་ཞིག་མི་ལུས་འདི་ལྟ་བུ་འཐོབ་པའི་ངེས་པ་མེད་དོ། །དེའི་ཕྱིར་དུ་བརྩོན་འགྲུས་ཆེན་པོས་དམ་པའི་ཆོས་ལ་སློབ་སྦྱོང་དང་ཉམས་ལེན་བྱེད་པ་ནི་ཤིན་ཏུ་གལ་ཆེའོ། །"""

print(f"\nИсходный текст (тибетский):\n{tibetan_text}")

# Токенизация текста
inputs = tokenizer(tibetan_text, return_tensors="pt", truncation=True, max_length=1024).to(device) # Добавил truncation и max_length на всякий случай

# Генерация перевода
# Код для русского языка: rus_Cyrl
# target_lang_code = "rus_Cyrl"
target_lang_code = "eng_Latn" # Новая строка для английского
target_lang_id = None

try:
    # Попытка получить ID языка с помощью convert_tokens_to_ids
    target_lang_id = tokenizer.convert_tokens_to_ids(target_lang_code)

    if target_lang_id is None or target_lang_id == tokenizer.unk_token_id:
        print(f"Предупреждение: Код языка '{target_lang_code}' не был распознан как известный токен напрямую "
              f"(ID: {target_lang_id}, UNK ID: {tokenizer.unk_token_id}).")
        # Альтернативная попытка, если язык является частью словаря напрямую (менее вероятно для NLLB-style кодов)
        # Эта проверка добавлена для полноты, но convert_tokens_to_ids должен быть основным методом для NLLB.
        if target_lang_code in tokenizer.get_vocab():
            target_lang_id = tokenizer.get_vocab()[target_lang_code]
            print(f"Найден ID через get_vocab(): {target_lang_id}")
        else:
            print(f"Код языка '{target_lang_code}' не найден и в get_vocab(). "
                  "Проверьте правильность кода или версию библиотеки transformers.")
            # Если все еще None или UNK, это проблема.
            if target_lang_id is None or target_lang_id == tokenizer.unk_token_id:
                 raise ValueError(f"Не удалось корректно определить ID для языка {target_lang_code}")
    
    print(f"ID для целевого языка ({target_lang_code}): {target_lang_id}")

except Exception as e:
    print(f"Критическая ошибка при получении ID для языка '{target_lang_code}': {e}")
    print("Возможные причины: неправильный код языка, несовместимая версия библиотеки transformers, "
          "или модель не поддерживает указанный язык в текущей конфигурации токенизатора.")
    print("Пожалуйста, убедитесь, что код языка корректен и токенизатор поддерживает его.")
    exit()

print("Генерация перевода...")
translated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=target_lang_id,
    # do_sample=True,
    max_length=512, 
    # temperature=0.5,         # Максимальная длина вывода
    num_beams=10,             # <<<--- ДОБАВЬТЕ ЭТОТ ПАРАМЕТР (например, 5 лучей)
    # early_stopping=True    # Можно добавить, чтобы остановить генерацию, когда все лучи закончат предложение
)

# Декодирование сгенерированных токенов в текст
russian_translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

print(f"\nМашинный перевод (русский):\n{russian_translation}")
