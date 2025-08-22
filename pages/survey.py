# pages/survey.py
from nicegui import ui
from state import app
from utils.save import save_user_survey

# 간단 저장소(없으면 생성)
if not hasattr(app.storage, 'surveys'):
    app.storage.surveys = {}  # { user_id: {'categories': [...], 'topic_text': '...', 'discomfort': int} }

CATEGORIES = ['학업', '진로/직업', '인간관계', '연애','정서 (우울 불안 분노 등)', '행동 및 습관', '가족', '경제적 불안 및 생활', '기타']

@ui.page('/survey/start')
def survey_start(room: str, mode: str, sid: str):
    """1페이지: 카테고리(복수 선택) + 자유 입력"""
    user_id = sid  # 별도 함수 없이 쿼리 파라미터 sid를 user_id로 사용

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
            categories = categories_select.value or []
            topic_text = (topic_area.value or '').strip()
            if not categories:
                error_label.text = '카테고리를 최소 1개 이상 선택해 주세요.'
                return
            if len(topic_text) < 5:
                error_label.text = '내용을 5자 이상 작성해 주세요.'
                return
            error_label.text = ''

            app.storage.surveys[user_id] = {
                'categories': categories,
                'topic_text': topic_text,
            }
            ui.navigate.to(f'/survey/discomfort?room={room}&mode={mode}&sid={sid}')
            ui.navigate.reload()

        with ui.row().classes('justify-end gap-2'):
            ui.button('다음으로', on_click=to_discomfort).props('color=primary')
            

@ui.page('/survey/discomfort')
def survey_discomfort(room: str, mode: str, sid: str):
    ui.label('사전 설문 (2/2)').classes('text-lg font-bold mb-1')

    with ui.card().classes('max-w-2xl w-full'):
        ui.label('SUS *').classes('text-lg font-bold mb-1')
        ui.label('현재 당신이 느끼는 주관적 불쾌감 정도를 0(전혀 불편하지 않음) ~ 5(극도로 불편함) 사이에서 평가해주십시오.') \
        .classes('text-gray-700 mb-4')
        # 상단: 0~5 숫자 라벨을 균등 간격으로


        # 중단: 라디오(0~5), 가로 배치
        with ui.row().classes('w-full'):
            scale = (
                ui.radio(options=[1,2,3,4,5], value=None)
                .props('inline')
                .classes('q-gutter-x-xl')         # 간격
                .style('text-align: center; width: 100%; display: block;')  # ★ 중앙정렬 핵심
            )
        # 하단: 양 끝 설명 라벨
        with ui.row().classes('w-full justify-between px-4 mt-2'):
            ui.label('전혀 불편하지 않음').classes('text-gray-600')
            ui.label('극도로 불편함').classes('text-gray-600')

        # --- 2) PANAS-NA (매트릭스 5점 척도) ---
        ui.label('PANAS-NA').classes('text-lg font-bold mt-6')
        ui.label('현재 당신이 느끼는 기분의 정도를 가장 잘 나타낸 곳에 표시해주십시오.') \
            .classes('text-gray-700 mb-2')

        LABEL_W = 160  # 왼쪽 문항 라벨 열 너비(px); 필요시 조절

        # 헤더: [공백 라벨열] + [옵션열(5개)]을 동일한 틀로
        with ui.element('div') \
            .classes('w-full') \
            .style(f'display:grid;grid-template-columns:{LABEL_W}px 1fr;gap:12px;align-items:center;'):
            ui.label('')  # 라벨열 자리 비움
            with ui.element('div') \
                .style('display:flex;justify-content:space-between;'):
                for h in ['전혀 그렇지 않다', '약간 그렇다 (2)', '보통 그렇다 (3)', '많이 그렇다 (4)', '매우 그렇다 (5)']:
                    ui.label(h).classes('text-gray-600')

        items = [
            '짜증난다', '긴장된다', '슬프다', '불안하다', '우울하다',
            '화가 난다', '짜증스럽다', '초조하다', '죄책감을 느낀다', '지친다'
        ]

        panas_values = {}  # {문항: 라디오컴포넌트}
        for text in items:
            with ui.element('div') \
                .classes('w-full') \
                .style(f'display:grid;grid-template-columns:{LABEL_W}px 1fr;gap:12px;align-items:center;'):
                ui.label(text)
                # 옵션열: flex로 균등 간격 정렬(헤더와 1:1 위치 대응)
                panas_values[text] = (
                    ui.radio(options=[1,2,3,4,5], value=None)
                      .props('inline size=md')
                      .classes('q-gutter-x-xl')
                      .style('display:flex;justify-content:space-between;width:100%;')
                )

        error = ui.label('').classes('text-red-500 text-sm mt-2')

        def go_prev():
            ui.navigate.to(f'/survey/start?room={room}&mode={mode}&sid={sid}')
            ui.navigate.reload()

        def go_next():
            if scale.value is None:
                error.text = '값을 선택해 주세요.'
                return
            error.text = ''
            user_id = app.storage.sessions.get(sid) or sid
            chat_id = app.storage.chat_rooms.get(user_id) or f'chatroom-{user_id}'
            app.storage.chat_rooms[user_id] = chat_id
            ui.navigate.to(f'/chat?room={chat_id}&mode={mode}&sid={sid}')
            ui.navigate.reload()

        with ui.row().classes('justify-between mt-4'):
            ui.button('이전으로', on_click=go_prev).props('outline')
            ui.button('다음으로', on_click=go_next).props('color=primary')

