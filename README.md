# review_sentiment_classification
네이버 영화 리뷰 데이터로 부터, 감정(Positive, Negative)을 분류하는 모델을 만드는 튜토리얼 <br>
html page: https://khlee88.github.io/review_sentiment_classification/

모델링 과정을 간단하게 요약면 다음과 같다.

1. 리뷰 단어들을 형태소 분석기를 통해 분리한다.
2. 단어들을 word2vec model을 통해 벡터화 한다.
3. RNN 기반의 모델을 통해 binomial classification 문제를 수행한다.
