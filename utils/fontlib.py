import matplotlib.pyplot as plt


def get_font_family():
    """
    시스템 환경에 따른 기본 폰트명을 반환하는 함수
    """

    import platform
    system_name = platform.system()

    if system_name == 'Darwin':
        font_family = "AppleGothic"
    elif system_name == "Windows":
        font_family = "Malgun Gothic"
    else:
        # Linux(Colab)
        # !apt-get install fonts-nanum -qq > /dev/null
        # !fc-cache -fv 

        import matplotlib as mpl
        mpl.font_manager._rebuild()
        findfont = mpl.font_manager.fontManager.findfont
        mpl.font_manager.findfont = findfont
        mpl.backends.backend_agg.findfont = findfont

        font_family = "NanumBarunGothic"
    return font_family


def set_font_family(font_family=get_font_family()):
    """
    matplotlib 폰트를 설정하는 함수
    style 설정은 꼭 폰트설정 위에서 합니다.
    style 에 폰트 설정이 들어있으면 한글폰트가 초기화 되어 한글이 깨집니다.
    """

    plt.style.use("seaborn")

    # 폰트 설정
    plt.rc("font", family=get_font_family())

    # 마이너스 폰트 설정
    plt.rc("axes", unicode_minus=False)

    # 그래프에 retina display 적용
    from IPython.display import set_matplotlib_formats
    set_matplotlib_formats("retina")
