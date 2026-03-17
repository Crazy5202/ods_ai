import pandas as pd

def make_features_evgeni(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['browser_language'])

    df["timezone"] = df["timezone"].fillna(-1).astype('int16')

    tmp = df[["event_id", "session_id", "event_dttm"]].copy()
    tmp = tmp.sort_values(["session_id", "event_dttm"])

    tmp["session_event_number"] = (
        (tmp.groupby("session_id").cumcount() + 1)
        .fillna(0)
        .astype("int16")
    )

    # номер операции внутри одной сессии
    df = df.merge(
        tmp[["event_id", "session_event_number"]],
        on="event_id",
        how="left"
    )

    df = df.drop(columns=['session_id'])

    df['operating_system_type'] = df['operating_system_type'].fillna(-1).astype('int8')

    # выбираем только % заряда
    df["battery"] = (
        df["battery"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .fillna(-1)
        .astype("int8")
    )

    # отбирает мажорную версию ОС
    df["device_system_version"] = (
        df["device_system_version"]
        .astype(str)
        .str.extract(r"^(\d+)")
        .astype("float32")
        .fillna(-1)
        .astype("int8")
    )

    size = df["screen_size"].astype(str).str.extract(r"(\d+)x(\d+)")

    w = pd.to_numeric(size[0], errors="coerce")
    h = pd.to_numeric(size[1], errors="coerce")

    # берем площадь экрана
    df["screen_size"] = (w * h).fillna(-1).astype("int32")

    df['developer_tools'] = df['developer_tools'].fillna(-1).astype('int8')
    df['phone_voip_call_state'] = df['phone_voip_call_state'].fillna(-1).astype('int8')
    df['web_rdp_connection'] = df['web_rdp_connection'].fillna(-1).astype('int8')
    df['compromised'] = df['compromised'].fillna(-1).astype('int8')

    return df