# pages/survey.py
from nicegui import ui
from state import app
from utils.save import save_user_survey

# 간단 저장소(없으면 생성)
if not hasattr(app.storage, 'surveys'):
    app.storage.surveys = {}  # { user_id: {'category': [...], 'topic_text': '...', 'discomfort': int} }

CATEGORIES = ['학업', '진로/직업', '인간관계', '연애','정서 (우울 불안 분노 등)', '행동 및 습관', '가족', '경제적 불안 및 생활', '기타']

@ui.page('/survey/start')
def survey_start(room: str, mode: str, sid: str):
    """1페이지: 카테고리(복수 선택) + 자유 입력"""
    user_id = app.storage.sessions.get(sid)

    ui.label('사전 설문 (1/2)').classes('text-2xl font-bold mb-4')
    with ui.card().classes('max-w-2xl w-full space-y-4'):

        ui.label('1) 대화 세션에서 이야기를 나누고 싶은 주제에 대해서 알려주세요.').classes('font-medium')
        categories_select = ui.select(
            CATEGORIES, multiple=False, with_input=False,
            label='카테고리',
        ).classes('w-full')

        ui.label('2) 관련해서 최근 1개월 이내에 경험하신 갈등이나 스트레스 상황에 대해 자세히 설명해 주세요.').classes('font-medium mt-2')
        topic_area = ui.textarea(
            label='지금 겪고 있는 상황, 느끼는 점, 기대하는 도움 등을 자유롭게 적어주세요. \n 어떤 일이었는지 자세하게 설명해 주실 수록 연구에 도움이 됩니다.',

        ).classes('w-full')

        error_label = ui.label('').classes('text-red-500 text-sm')

        def to_discomfort():
            category = (categories_select.value or '').strip()
            topic_text = (topic_area.value or '').strip()
            if not category:
                error_label.text = '카테고리를 선택해 주세요.'
                return
            if len(topic_text) < 5:
                error_label.text = '내용을 5자 이상 작성해 주세요.'
                return
            error_label.text = ''

            app.storage.surveys[user_id] = {
                'category': category,
                'topic_text': topic_text,
            }
            ui.navigate.to(f'/survey/discomfort?room={room}&mode={mode}&sid={sid}')
            ui.navigate.reload()

        with ui.row().classes('justify-end gap-2'):
            ui.button('다음으로', on_click=to_discomfort).props('color=primary')
            

@ui.page('/survey/discomfort')
def survey_discomfort(room: str, mode: str, sid: str):
    user_id = app.storage.sessions.get(sid)
    if not user_id:
        ui.notify('세션이 만료되었습니다. 다시 시작해 주세요.', type='negative')
        return

    if not hasattr(app.storage, 'surveys'):
        app.storage.surveys = {}

    base = app.storage.surveys.get(user_id, {})  # 1페이지 값/이전 저장값

    ui.label('사전 설문 (2/2)').classes('text-lg font-bold mb-1')

    with ui.scroll_area().style('height: calc(100vh - 120px); width: 100%;'):
        with ui.column().classes('items-center w-full py-4'):
            with ui.card().classes('max-w-2xl w-full').style('overflow: visible;'):

                # === SUDS (0~100) ===
                ui.label('SUDS').classes('text-lg font-bold mb-1')
                ui.label('현재 느끼는 주관적 불쾌감(0=전혀 불편하지 않음, 100=극도로 불편함)을 선택해 주세요.') \
                    .classes('text-gray-700 mb-4')

                with ui.element('div').style(
                    'display:grid;grid-template-columns:40px 1fr 52px 44px;'
                    'align-items:center;gap:12px;overflow:visible;'
                    'padding-right:12px;width:100%;'
                ):
                    ui.label('0').classes('text-gray-600 text-right')
                    suds_slider = (
                        ui.slider(min=0, max=100,
                                  value=base.get('discomfort') if isinstance(base.get('discomfort'), int) else None,
                                  step=1)
                          .props('markers')
                          .classes('w-full')
                          .style('min-width:0;')
                    )
                    ui.label('—') \
                        .bind_text_from(suds_slider, 'value',
                                        lambda v: f'{v}' if v is not None else '—') \
                        .classes('text-gray-700 text-right')
                    ui.label('100').classes('text-gray-600')

                ui.separator().classes('my-4')

                # === 통제감 (0~3) ===
                ui.label('통제감').classes('text-lg font-bold mb-1')
                ui.label('그 상황에 대해 당신은 얼마나 통제할 수 있다고 느끼셨습니까?') \
                    .classes('text-gray-700 mb-2')

                LABEL_W = 160
                # 헤더: [공백] + [옵션열(4개)] / "텍스트 (숫자)" 형식
                with ui.element('div') \
                    .classes('w-full') \
                    .style(f'display:grid;grid-template-columns:{LABEL_W}px 1fr;gap:12px;align-items:center;'):
                    ui.label('')
                    with ui.element('div') \
                        .style('display:flex;justify-content:space-between;'):
                        for h in [
                            '전혀 통제할 수 없었다 (0)',
                            '약간 통제할 수 있었다 (1)',
                            '어느 정도 통제할 수 있었다 (2)',
                            '많이 통제할 수 있었다 (3)',
                        ]:
                            ui.label(h).classes('text-gray-600')

                # 라디오: 값은 int로 저장/복원
                with ui.element('div') \
                    .classes('w-full') \
                    .style(f'display:grid;grid-template-columns:{LABEL_W}px 1fr;gap:12px;align-items:center;'):
                    ui.label('통제감')
                    control_radio = (
                        ui.radio(
                            options=[0, 1, 2, 3],
                            value=base.get('control') if isinstance(base.get('control'), int) else None
                        )
                        .props('inline size=md')
                        .classes('q-gutter-x-xl')
                        .style('display:flex;justify-content:space-between;width:100%;')
                    )

                ui.separator().classes('my-4')
                # === 중요도 (1~6) ===
                ui.label('중요도').classes('text-lg font-bold mb-1')
                ui.label('전체적인 맥락에서 볼 때, 이 사건은 당신에게 얼마나 중요한 일이었습니까?') \
                    .classes('text-gray-700 mb-2')

                # ✅ 헤더: grid로 6칸 균등 분할 + 줄바꿈 허용(칸 밖 방지)
                with ui.element('div') \
                    .classes('w-full') \
                    .style(f'display:grid;grid-template-columns:{LABEL_W}px 1fr;gap:12px;align-items:start;'):
                    ui.label('')  # 라벨 열 자리
                    with ui.element('div') \
                        .style('display:grid;grid-template-columns:repeat(6, minmax(0, 1fr));gap:8px;'):
                        for h in [
                            '전혀 중요하지 않았다 (1)',
                            '조금 중요했다 (2)',
                            '다소 중요했다 (3)',
                            '보통 정도로 중요했다 (4)',
                            '매우 중요했다 (5)',
                            '극도로 중요했다 (6)',
                        ]:
                            ui.label(h) \
                            .classes('text-gray-600 text-center') \
                            .style('white-space:normal;overflow-wrap:anywhere;word-break:break-word;min-width:0;')

                # ✅ 라디오: 동일한 6칸 그리드 위에 정중앙 배치
                with ui.element('div') \
                    .classes('w-full') \
                    .style(f'display:grid;grid-template-columns:{LABEL_W}px 1fr;gap:8px;align-items:center;'):
                    ui.label('중요도')
                    importance_radio = (
                        ui.radio(
                            options=[1, 2, 3, 4, 5, 6],
                            value=base.get('importance') if isinstance(base.get('importance'), int) else None
                        )
                        .classes('w-full')
                        # QOptionGroup 루트에 grid를 적용해서 아이템을 6칸에 배치
                        .style('display:grid;grid-template-columns:repeat(6, minmax(0, 1fr));gap:8px;justify-items:center;')
                    )


                ui.separator().classes('my-4')
                # === PANAS-NA (10문항, 5점) ===
                ui.label('PANAS-NA').classes('text-lg font-bold mb-1')
                ui.label('현재 느끼는 기분의 정도를 가장 잘 나타내는 곳에 표시해 주세요.') \
                    .classes('text-gray-700 mb-2')

                with ui.element('div') \
                    .classes('w-full') \
                    .style(f'display:grid;grid-template-columns:{LABEL_W}px 1fr;gap:12px;align-items:center;'):
                    ui.label('')
                    with ui.element('div') \
                        .style('display:flex;justify-content:space-between;'):
                        for h in ['전혀 그렇지 않다 (1)', '약간 그렇다 (2)', '보통 그렇다 (3)', '많이 그렇다 (4)', '매우 그렇다 (5)']:
                            ui.label(h).classes('text-gray-600')

                items = [
                    '짜증난다', '긴장된다', '슬프다', '불안하다', '우울하다',
                    '화가 난다', '짜증스럽다', '초조하다', '죄책감을 느낀다', '지친다'
                ]
                prev_panas = base.get('panas_na') if isinstance(base.get('panas_na'), dict) else {}
                panas_values = {}
                for text in items:
                    with ui.element('div') \
                        .classes('w-full') \
                        .style(f'display:grid;grid-template-columns:{LABEL_W}px 1fr;gap:12px;align-items:center;'):
                        ui.label(text)
                        panas_values[text] = (
                            ui.radio(options=[1,2,3,4,5],
                                     value=prev_panas.get(text) if isinstance(prev_panas.get(text), int) else None)
                              .props('inline size=md')
                              .classes('q-gutter-x-xl')
                              .style('display:flex;justify-content:space-between;width:100%;')
                        )

                error = ui.label('').classes('text-red-500 text-sm mt-2')

                def go_prev():
                    # 현재 입력 임시 저장 후 이전 페이지로
                    temp = {
                        'discomfort': suds_slider.value if suds_slider.value is not None else base.get('discomfort'),
                        'control': control_radio.value if control_radio.value is not None else base.get('control'),
                        'importance': importance_radio.value if importance_radio.value is not None else base.get('importance'),
                        'panas_na': {
                            k: int(comp.value) for k, comp in panas_values.items() if comp.value is not None
                        } or prev_panas
                    }
                    app.storage.surveys[user_id] = {**base, **temp}
                    ui.navigate.to(f'/survey/start?room={room}&mode={mode}&sid={sid}')

                def go_next():
                    # 검증
                    if suds_slider.value is None:
                        error.text = '불쾌감(SUDS)을 선택해 주세요.'
                        return
                    if control_radio.value is None:
                        error.text = '통제감을 선택해 주세요.'
                        return
                    if importance_radio.value is None:
                        error.text = '중요도를 선택해 주세요.'
                        return
                    for k, comp in panas_values.items():
                        if comp.value is None:
                            error.text = f'PANAS-NA: "{k}" 항목을 선택해 주세요.'
                            return
                    error.text = ''

                    payload = {
                        'category': base.get('category', ''),
                        'topic_text': base.get('topic_text', ''),
                        'discomfort': int(suds_slider.value),           # 0~100
                        'control': int(control_radio.value),            # 0~3
                        'importance': int(importance_radio.value),      # 1~6
                        'panas_na': {k: int(comp.value) for k, comp in panas_values.items()},
                    }

                    # 임시 저장
                    app.storage.surveys[user_id] = {**base, **payload}

                    # 영속 저장
                    save_user_survey(user_id, mode, payload)

                    # 채팅방 이동
                    chat_id = app.storage.chat_rooms.get(user_id) or f'chatroom-{user_id}'
                    app.storage.chat_rooms[user_id] = chat_id
                    ui.navigate.to(f'/chat?room={chat_id}&mode={mode}&sid={sid}')
                    ui.navigate.reload() 

                with ui.row().classes('justify-between mt-4'):
                    ui.button('이전으로', on_click=go_prev).props('outline')
                    ui.button('다음으로', on_click=go_next).props('color=primary')



