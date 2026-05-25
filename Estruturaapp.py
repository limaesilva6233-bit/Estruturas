import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from anastruct.fem.system import SystemElements

st.set_page_config(page_title="MRD Scanner Pro", layout="wide")

st.title("📐 Scanner Estrutural Hiperestático (MRD + OpenCV)")
st.write("Insira a foto do caderno. O sistema lerá os nós (amarelo) e gerará a estrutura para o cálculo.")

# --- SIDEBAR: ENTRADAS DE MATERIAIS ---
with st.sidebar:
    st.header("1. Propriedades Mecânicas")
    arquivo_imagem = st.file_uploader("Suba a foto do sistema:", type=["png", "jpg", "jpeg"])
    
    st.markdown("---")
    E_gpa = st.number_input("Módulo de Elasticidade E (GPa)", value=210)
    A_cm2 = st.number_input("Área da Seção A (cm²)", value=50)
    I_cm4 = st.number_input("Momento de Inércia I (cm⁴)", value=2000)
    
    # Conversões para Sistema Internacional (kN, m)
    E_val = E_gpa * 1e6
    A_val = A_cm2 / 10000
    I_val = I_cm4 / 1e8

# --- DETECÇÃO REAL DE CORES E GEOMETRIA ---
if arquivo_imagem is not None:
    file_bytes = np.asarray(bytearray(arquivo_imagem.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    
    # Máscara para Nós (Amarelo) - Ajustada para ser um pouco mais restrita
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([32, 255, 255])
    mask_nos = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # Encontrar os nós amarelos na foto
    contours_nos, _ = cv2.findContours(mask_nos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lista_nos = []
    
    for cnt in contours_nos:
        # Aumentamos o limite de área de 30 para 300 para ignorar as setas finas da carga distribuída
        if cv2.contourArea(cnt) > 300:  
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m20"] / M["m00"])
                lista_nos.append((cx, cy))
                
    # Ordenar nós
    lista_nos = sorted(lista_nos, key=lambda x: (x[1], x[0])) # Ordena por altura e depois largura
    num_nos = len(lista_nos)
    
    barras_detectadas = []
    if num_nos > 1:
        # Cria um encadeamento simples baseado no número de nós
        for i in range(num_nos - 1):
            barras_detectadas.append((f"Nó {i+1}", f"Nó {i+2}"))

    # --- CORPO DA INTERFACE ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔍 Elementos Mapeados na Foto")
        
        img_feedback = img.copy()
        for i, (cx, cy) in enumerate(lista_nos):
            cv2.circle(img_feedback, (cx, cy), 20, (0, 0, 255), 3) 
            cv2.putText(img_feedback, f"No {i+1}", (cx - 25, cy - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
        st.image(img_feedback, channels="BGR", use_column_width=True)
        st.success(f"O OpenCV identificou: {num_nos} Nós Principais.")

    with col2:
        st.subheader("⚙️ Calibração Manual Baseada no Desenho")
        
        if num_nos < 2:
            st.error("Ajuste a iluminação ou o tamanho do desenho. O sistema precisa detectar os círculos amarelos maiores.")
        else:
            comprimentos = {}
            st.markdown("#### Comprimento das Barras (m)")
            for i, barra in enumerate(barras_detectadas):
                comprimentos[i] = st.number_input(
                    f"Comprimento do {barra[0]} até o {barra[1]}:", 
                    value=4.0, 
                    min_value=0.1, 
                    key=f"len_{i}"
                )
            
            st.markdown("---")
            st.markdown("#### Forças Concentradas (Seta Verde)")
            no_carga = st.selectbox("Nó da Força Concentrada:", [f"Nó {i+1}" for i in range(num_nos)], index=1 if num_nos>1 else 0)
            val_fx = st.number_input(f"Força Horizontal Fx no {no_carga} (kN):", value=10.0)
            val_fy = st.number_input(f"Força Vertical Fy no {no_carga} (kN):", value=0.0)

            st.markdown("---")
            st.markdown("#### Carga Distribuída")
            tem_distribuida = st.checkbox("Existe carga distribuída em alguma barra?", value=True)
            val_q = 0.0
            barra_com_carga = 0
            if tem_distribuida:
                barra_com_carga = st.selectbox("Selecione a Barra com carga distribuída:", [f"Barra {i+1}" for i in range(len(barras_detectadas))])
                val_q = st.number_input("Valor da Carga Distribuída (kN/m) [Sinal Negativo = Para baixo]:", value=-10.0)

    # --- MOTOR DE CÁLCULO (ANASTUCT) ---
    if num_nos >= 2:
        st.markdown("---")
        if st.button("🚀 Resolver Estrutura via Rigidez Direta"):
            try:
                ss = SystemElements()
                
                # Para o seu pórtico de 4 nós:
                # Vamos definir uma lógica de pórtico simples caso ache 4 nós:
                if num_nos == 4:
                    # Coordenadas reais calculadas com base nos inputs manuais
                    # Nó 1 (Apoio Esquerdo Base), Nó 2 (Quina Esquerda), Nó 3 (Quina Direita), Nó 4 (Apoio Direito Base)
                    h1 = comprimentos[0] # Altura pilar esquerdo
                    L_viga = comprimentos[1] # Comprimento da viga
                    h2 = comprimentos[2] # Altura pilar direito
                    
                    coords = {
                        1: [0.0, 0.0],
                        2: [0.0, h1],
                        3: [L_viga, h1],
                        4: [L_viga, h1 - h2]
                    }
                    
                    # Adicionar Elementos
                    ss.add_element(location=[coords[1], coords[2]], EA=E_val*A_val, EI=E_val*I_val)
                    ss.add_element(location=[coords[2], coords[3]], EA=E_val*A_val, EI=E_val*I_val)
                    ss.add_element(location=[coords[3], coords[4]], EA=E_val*A_val, EI=E_val*I_val)
                    
                    # Apoios (Azuis no seu desenho)
                    ss.add_support_fixed(node_id=1) # Engaste na esquerda
                    ss.add_support_fixed(node_id=4) # Engaste na direita (ou troque por hinged se for apoio de 2º gênero)
                    
                else:
                    # Lógica linear padrão para outros casos
                    coords = {1: [0.0, 0.0]}
                    for i in range(len(barras_detectadas)):
                        coords[i+2] = [coords[i+1][0] + comprimentos[i], 0.0]
                        ss.add_element(location=[coords[i+1], coords[i+2]], EA=E_val*A_val, EI=E_val*I_val)
                    ss.add_support_fixed(node_id=1)
                    ss.add_support_hinged(node_id=num_nos)

                # Aplicar Força Concentrada corrigida (add_point_load)
                idx_no_carga = int(no_carga.split(" ")[1])
                if val_fx != 0 or val_fy != 0:
                    ss.add_point_load(node_id=idx_no_carga, Fx=val_fx, Fy=val_fy)
                
                # Aplicar Carga Distribuída se marcado
                if tem_distribuida and val_q != 0:
                    idx_barra = int(barra_com_carga.split(" ")[1])
                    ss.add_distributed_load(element_id=idx_barra, q=val_q)
                
                # Solucionar
                ss.solve()
                
                # --- PLOTAGEM DOS DIAGRAMAS ---
                st.subheader("📊 Diagramas de Esforços Solicitantes")
                g1, g2, g3 = st.columns(3)
                
                with g1:
                    st.write("**Momento Fletor (M)**")
                    fig_m, ax_m = plt.subplots()
                    ss.plot_bending_moment(show=False, ax=ax_m)
                    st.pyplot(fig_m)
                with g2:
                    st.write("**Esforço Cortante (V)**")
                    fig_v, ax_v = plt.subplots()
                    ss.plot_shear_force(show=False, ax=ax_v)
                    st.pyplot(fig_v)
                with g3:
                    st.write("**Esforço Axial (N)**")
                    fig_n, ax_n = plt.subplots()
                    ss.plot_axial_force(show=False, ax=ax_n)
                    st.pyplot(fig_n)

            except Exception as erro:
                st.error(f"Erro na montagem da matriz de rigidez: {erro}")
