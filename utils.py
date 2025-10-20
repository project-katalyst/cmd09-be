from ast import literal_eval
import json
from urllib.parse import urlparse
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name
from scrapping import collect_home_links, get_relevant_contents, clean_pages
from ai import get_relevant_links, get_tags, summarize_company
from typing import Tuple, cast


TAGS_DF_UN = pd.read_csv('./dados/tags/unspsc+.csv')
BUYERS_FINANCIALS = pd.read_csv('./dados/buyers/companies_financials.csv')
BUYERS_DF = pd.read_parquet('./dados/buyers/buyers_final.parquet', engine='fastparquet')


def get_contents(url: str, model) -> Tuple[dict[str, str], int, int]:
    links, _ = collect_home_links(url)
    relevant_links = get_relevant_links(links)
    relevant_pages_contents = get_relevant_contents(
        relevant_links)
    if len(relevant_pages_contents) == 0:
        print('All links invalid.')
        return {'': ''}, 0, 0

    return clean_pages(relevant_pages_contents)


def get_company_tags(summary: str, df: pd.DataFrame, company_name: str, model='gpt-4o-mini'):
    possible_tags = {}
    all_tags = {}

    depth = df.shape[1]
    tags = {}

    for level in range(1, depth):
        if level == 1:
            all_tags[level] = list(df['tag1'].unique())
        else:
        #     possible_tags[level-1] = [value for key, value in tags.items() if key.count('.') == (level-2)]

            if not possible_tags.get(level - 1):
                break

            all_tags[level] = []
            for prev_tag in possible_tags[level - 1]:
                subset = df[df[f'tag{level - 1}'] == prev_tag]
                all_tags[level].extend(list(subset[f'tag{level}'].unique()))

            all_tags[level] = list(set(all_tags[level]))

        if len(all_tags[level]) == 0:
            break

        possible_tags[level] = get_tags(all_tags[level], summary, level, possible_tags[level-1] if level != 1 else {}, company_name, model)


        if len(possible_tags[level]) == 0:
            break

        for selected_tag in possible_tags[level]:
            try:
                code_val = str(df[df[f'tag{level}'] == selected_tag]['code'].iloc[0])
                tags[code_val] = selected_tag
            except IndexError:
                print(f'{selected_tag} not found at level {level}')
            except ValueError:
                print(f'{selected_tag} has invalid code at level {level}')
        

    return tags


def tag_selection(selection_per_level: dict, tags_df: pd.DataFrame, verbose=True) -> dict:
    tags = {}
    for level in range(1, tags_df.shape[1]):

        if level in selection_per_level.keys() and len(selection_per_level[level]) == 0:
            break

        try:
            s_per_level = selection_per_level[level]
        except KeyError:
            s_per_level = selection_per_level[str(level)]

        for selected_tag in s_per_level:
            try:
                code_val = tags_df[tags_df[f'tag{level}']
                                   == selected_tag]['code'].iloc[0]
                tags[code_val] = selected_tag
            except IndexError:
                if verbose:
                    print(f'{selected_tag} not found at level {level}')
            except ValueError:
                if verbose:
                    print(f'{selected_tag} has invalid code at level {level}')

    return tags

def get_tag_code(tag: str, tags_df: pd.DataFrame) -> str:
    try:
        code_val = tags_df[tags_df.isin([tag]).any(axis=1)]['code'].iloc[0]
        return code_val
    except IndexError:
        print(f'{tag} not found in tags dataframe')
        return ''
    except ValueError:
        print(f'{tag} has invalid code in tags dataframe')
        return ''


def remove_ancestors(codes_dict) -> dict[str, str]:
    if type(codes_dict) is dict:
        codes = list(codes_dict.keys())
    else:
        codes = codes_dict
    ancestors = set()

    for c1 in codes:
        for c2 in codes:

            if c1 != c2 and c2.startswith(c1 + "."):
                ancestors.add(c1)

    final_codes = [c for c in codes if c not in ancestors and c.count('.') > 1]

    if type(codes_dict) is dict:
        final_codes = {c: codes_dict[c] for c in final_codes}
    return final_codes


def build_codes_df(codes:list[str], tags_df:pd.DataFrame) -> pd.DataFrame:

    rows = []
    depth = tags_df.shape[1]-1

    for full_code in codes:
        parts = full_code.split(".")

        row_data = {f"tag{i+1}": "" for i in range(depth)}
        row_data["code"] = full_code

        for level_idx in range(1, len(parts)+1):
            partial_code = ".".join(parts[: level_idx])
            row_data[f"tag{level_idx}"] = tags_df[tags_df['code']
                                                  == partial_code][f'tag{level_idx}'].iloc[0]

        rows.append(row_data)

    df_tags = pd.DataFrame(
        rows,
        columns=[f"tag{i}" for i in range(1, depth + 1)] + ["code"]
    )

    df_tags = (
        df_tags
        .sort_values(by=[f'tag{i}' for i in range(1, depth)])
        .drop(columns=["code"])
        .reset_index(drop=True)
    )

    return df_tags


def df_to_codes_dict(df: pd.DataFrame, tags_df: pd.DataFrame, verbose: bool = False):
    tags = {}
    depth = tags_df.shape[1]
    for row in df.iterrows():
        for level in range(1, depth):
            selected_tag = row[1][f'tag{level}']
            try:
                code_val = tags_df[tags_df[f'tag{level}']== selected_tag]['code'].iloc[0]
                tags[code_val] = selected_tag
            except IndexError:
                if verbose:
                    print(f'{selected_tag} not found at level {level}')
            except ValueError:
                if verbose:
                    print(f'{selected_tag} has invalid code at level {level}')
    return tags


def get_sellside_tags(url: str, model='gpt-4o-mini') -> Tuple[pd.DataFrame, str]:
    company = urlparse(url).netloc
    print(f'\n{30*"="} {company} {30*"="}\n')
    contents = get_contents(
        url, model)
    company_summary = summarize_company(
        contents, model)

    tags_un = get_company_tags(company_summary, TAGS_DF_UN, company)

    return build_codes_df(list(remove_ancestors(tags_un).keys()), tags_df=TAGS_DF_UN), company_summary


# def matching(target_dfs: list[pd.DataFrame], buyer_dfs: list[pd.DataFrame]) -> list[float]:
#     depths = [10, 6]
#     scores = [0.0, 0.0]
#     dfs = [df.copy() for df in target_dfs]
#     possible_weights = [[30, 25, 20, 12, 4, 4, 3, 1, 1], [37, 27, 17, 12, 7]]

#     for i in range(2):
#         weights = possible_weights[i]
#         df = dfs[i]

#         for j in range(1, depths[i]):
#             weight = weights[j-1]
#             column = f'tag{j}'

#             for tag in df[df[column] != ''][column]:

#                 if buyer_dfs[i][buyer_dfs[i][column] == tag].shape[0] > 0:

#                     scores[i] += weight/len(df[df[column] != ''][column])

#             df = df[df[column].isin(list(buyer_dfs[i][column].unique()))]
#             df = df[df[f'tag{j+1}'] != ''] if j < (depths[i]-1) else df
#             df = df.drop(columns=[column])

#     return round(max(scores), 1)
       

def matching_by_code(target_tags: list[str], buyer_dfs: list[str]):
    depth = 6
    score=0
    weights = [7, 12, 17, 27, 37]

    # for i in range(2):
    target_codes = [tag.split('.') for tag in target_tags]
    buyer_codes = [tag.split('.') for tag in buyer_dfs]

    for j in range(0, depth-1):

        weight = weights[j]
        remove_idxs = []
        level_codes = ['.'.join(code[:j+1]) for code in target_codes if len(code) > j]

        for pos, code in enumerate(level_codes):
            if code in ['.'.join(i[:j+1]) for i in buyer_codes]:
                score += weight/len(level_codes)
            else:
                remove_idxs.append(pos)
        remove_idxs.sort(reverse=True)
        for idx in remove_idxs:
            target_codes.pop(idx)

    return score

def get_financials(tiker:str) -> list[str,dict[str,float]]:
    ticker = tiker.lower()
    try:
        ebitda = BUYERS_FINANCIALS[(BUYERS_FINANCIALS['ticker'] == ticker)&((BUYERS_FINANCIALS['nome_coluna']=='EBITDA')|(BUYERS_FINANCIALS['nome_coluna']=='Dívida líquida'))][['nome_coluna','valor','data']]
        if ebitda.empty:
            return ['EBITDA não encontrado',0.0]
    except IndexError:
        return ['EBITDA não encontrado',0.0]
    mais_recente = ebitda[ebitda['data'].str.contains('2025')][['nome_coluna','valor','data']]
    if mais_recente.empty:
        mais_recente = ebitda[ebitda['data'].str.contains('2024')].sort_values(by='data', ascending=True)[['nome_coluna','valor','data']]

    return [mais_recente.iloc[-1]['data'], {'EBITDA':float(mais_recente[mais_recente['nome_coluna']=='EBITDA'].iloc[-1]['valor']), 'Dívida líquida':float(mais_recente[mais_recente['nome_coluna']=='Dívida líquida'].iloc[-1]['valor'])}]

def get_lowest_multiple(sectors: list[str]) -> float:
    lowest = float('inf')
    with open('./dados/buyers/multiplos.json', 'r', encoding='utf-8') as f:
        multiples = json.loads(f.read())
    for multiplo in multiples:
        if get_tag_code(multiplo['sector'], TAGS_DF_UN) in sectors:
            if multiplo['multiple'] < lowest:
                lowest = multiplo['multiple']
    return lowest if lowest != float('inf') else 0.0


def calculate_merger_net_debt(buyer_ebitda: float, buyer_dl: float, target_ebitda: float, multiple: float) -> float:
    return (buyer_dl+ (target_ebitda * multiple))/(buyer_ebitda + target_ebitda)

def convert_input(endpoint_input: dict[str, list[str]]) -> list[str]:
    tags_codes = []
    for item in endpoint_input.values():
        for tag in item:
            if tag != '':
                tags_codes.append(get_tag_code(tag, TAGS_DF_UN))
    return remove_ancestors(tags_codes)

def matching(target_tags: dict[str, list[str]], target_ebitda: float) -> Tuple[list[dict], float]:
    target_tags = convert_input(target_tags)
    target_sectors = set([tag.split('.')[0] for tag in target_tags])
    scores = []
    for buyer in BUYERS_DF.itertuples():
        if target_sectors & set(buyer[-1]):
            data, financials = get_financials(buyer[2])
            buyer_ebitda = financials['EBITDA']
            buyer_dl = financials['Dívida líquida']
            multiple = get_lowest_multiple(target_sectors)
            if calculate_merger_net_debt(buyer_ebitda, buyer_dl, target_ebitda, multiple) < 3.5:
                score = matching_by_code(target_tags, buyer[5])
                scores.append({'Nome': buyer[1], 'Site': buyer[3], 'Resumo': buyer[4], 'Score': round(score,1), 'EBITDA': buyer_ebitda, 'Data do EBITDA': data})
            else:
                print(f"Buyer {buyer[1]} skipped due to high net debt after merger.")
    return scores, target_ebitda * multiple

def get_financials_report(tiker:str, data:str) -> pd.DataFrame:
    ticker = tiker.lower()
    try:
        financials = BUYERS_FINANCIALS[(BUYERS_FINANCIALS['ticker'] == ticker)&(BUYERS_FINANCIALS['data'].str.contains(data))]
        if financials.empty:
            return pd.DataFrame()
    except IndexError:
        return pd.DataFrame()
    return financials


def generate_target_list(company_tags:list[str], buyers: pd.DataFrame, n_results: int = 20, output_file: str = 'output.xlsx') -> pd.DataFrame:
    scores = []

    for row in buyers.itertuples():
        score = matching_by_code(company_tags, row[3])
        
        scores.append({'Nome': row[1], 'Score': round(score/100,2), 'Resumo': row[2], 'URL': row[4]})
    
    
    tl = pd.DataFrame(scores)
    tl = tl.sort_values(by='Score', ascending=False).reset_index(drop=True)
    return tl

def export_to_excel(df: pd.DataFrame, file_name: str):
    """
    Exports a DataFrame to an Excel file with formatted sheets.

    Args:
        df (pd.DataFrame): The DataFrame to export.
        file_name (str): The name of the output Excel file.
    """
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Resultados', index=False)
        format_sheet(df, writer, 'Resultados')

def format_sheet(df: pd.DataFrame, writer: pd.ExcelWriter, sheet_name: str):

    workbook = cast(xlsxwriter.Workbook, writer.book)
    worksheet = writer.sheets[sheet_name]

    wrap_format = workbook.add_format({'text_wrap': True})
    worksheet.set_column('A:A', 60)
    worksheet.set_column('B:B', 5)
    worksheet.set_column('C:C', 100, wrap_format)
    worksheet.set_column('D:D', 40)
    worksheet.set_column('E:E', 15)
    worksheet.set_column('F:F', 15)
    worksheet.set_column('G:G', 15)
    score_col_idx = df.columns.get_loc('Score')

    score_col_letter = xl_col_to_name(score_col_idx)

    score_range = f'{score_col_letter}2:{score_col_letter}{len(df) + 1}'

    worksheet.conditional_format(score_range, {
        'type': '2_color_scale',
        'min_color': "#FFEB84",
        'max_color': "#63BE7B"
    })

    worksheet.freeze_panes(1, 0)


def gerar_relatorio_md(nome_arquivo, empresas, matchs):
    """
    Gera um arquivo .md com os resumos, tags e resultados dos matchs entre empresas.

    Args:
        nome_arquivo (str): Caminho do arquivo .md de saída.
        empresas (list): Lista de dicts, cada um com 'nome', 'resumo', 'tags'.
        matchs (list): Lista de tuplas (nome1, nome2, score).
    """
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Tags e Match de Empresas\n\n")
        f.write("## 1. Tags e Resumo das Empresas\n\n")
        for empresa in empresas:
            f.write(f"### {empresa['nome']}\n\n")
            f.write(f"**Resumo:**\n{empresa['resumo']}\n\n")
            f.write(f"**Tags:**\n{(empresa['tags'])}\n\n---\n\n")
        f.write("## 2. Resultados dos Matchs\n\n")
        for nome1, nome2, score in matchs:
            f.write(f"- **Match {nome1} x {nome2}:** {score}\n")

    
