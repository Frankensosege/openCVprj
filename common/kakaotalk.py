import requests
import json

'''
파이썬 코드를 통해 카카오톡에 메시지를 보낼 수 있는 인증 토큰을 발급발고 그 토큰을
이용해 카카오톡에 메시지를 보내보자.
'''
class KatalkApi:
    ### 인증 토큰을 받아 json 형태로 저장
    def __init__(self):
        self.url = 'https://kauth.kakao.com/oauth/token'
        self.client_id = 'fc175fc626464cb927d5eba82088378f'
        self.redirect_uri = 'https://example.com/oauth'
        self.code = 'cY_36wEDPJBnehQsJqAI3ZOodZ2ZX26tCeAlxaYQpjfbAls-SNwHmq9_LoyGXrHJFm1R8woqJVMAAAGHnS3AhQ'
        # self.code = token

    def getJson(self):
        data = {
            'grant_type':'authorization_code',
            'client_id':self.client_id,
            'redirect_uri':self.redirect_uri,
            'code': self.code,
            }

        response = requests.post(self.url, data=data)
        tokens = response.json()

        #발행된 토큰 저장
        with open("./data/kakao/token.json","w") as kakao:
            json.dump(tokens, kakao)


    ## 저장되어 있는 토큰 정보를 가지고 메시지를 보내기
    def sendMessage(self, message):
        # 발행한 토큰 불러오기
        with open("./data/kakao/token.json", "r") as kakao:
            tokens = json.load(kakao)

        url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

        headers = {
            "Authorization": "Bearer " + tokens["access_token"]
        }

        data = {
            'object_type': 'text',
            'text': message,
            'link': {
                'web_url': 'https://developers.kakao.com',
                'mobile_web_url': 'https://developers.kakao.com'
            }
        }

        data = {'template_object': json.dumps(data)}
        response = requests.post(url, headers=headers, data=data)
        response.status_code