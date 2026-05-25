import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from anastruct.fem.system import SystemElements

st.set_page_config(page_title="MRD Scanner Pro", layout="wide")

st.title("📐 Scanner Estrutural Hiperestático (MRD + Controle Manual)")
st.write("Suba a foto do caderno. O sistema tentará mapear a geometria, e você poderá ajustar os nós e barras livremente se houver erros de leitura.")

# --- SIDEBAR: PROPRIEDADES E UPLOAD ---
with st.sidebar:
    st.header("1. Parâmetros do Sistema")
    arquivo_imagem = st.file_uploader("Suba a foto do sistema:", type=["png", "jpg", "jpeg"])
    
    st.markdown("---")
    E_gpa = st.number_input("Módulo de Elasticidade E (GPa)", value=210)
    A_cm2 = st.number_input("Área da Seção A (cm²)", value=50)
    I_cm4 = st.number_input("Momento de Inércia I (cm⁴)", value=2000)
    
    # Conversões para o Sistema Internacional (kN, m)
    E_val = E_gpa * 1e6
    A_val = A_cm2 / 10000
    I_val = I_cm4 / 1e8

# --- PROCESSAMENTO DA IMAGEM (OPENCV) ---
num_nos_sugeridos = 4 # Valor padrão caso não tenha imagem

if arquivo_imagem is not None:
    file_bytes = np.asarray(bytearray(arquivo_imagem.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    
    # Filtro para os nós amarelos
    yellow_lower = np.array([15, 80, 80])
    yellow_upper = np.array([35, 255, 255])
    mask_nos = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    contours_nos, _ = cv2.findContours(mask_nos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lista_nos = []
    
    for cnt in contours_nos:
        if cv2.contourArea(cnt) > 250: # Filtra pontos muito pequenos
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m20"] / M["m00"])
                lista_nos.append((cx, cy))
                
    if len(lista_nos) >= 2:
        num_nos_sugeridos = len(lista_nos)

# --- CORPO PRINCIPAL DA INTERFACE ---
if arquivo_imagem is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔍 Visualização do Scanner")
        
        # Feedback visual na tela
        img_feedback = img.copy()
        if 'lista_nos' in locals() and len(lista_nos) > 0:
            for i, (cx, cy) in enumerate(sorted(lista_nos, key=lambda x: (x[1], x[0]))):
                cv2.circle(img_feedback, (cx, cy), 20, (0, 0, 255), 3)
                cv2.putText(img_feedback, f"Detectado", (cx - 40, cy - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        st.image(img_feedback, channels="BGR", use_column_width=True)
        st.info(f"O OpenCV sugeriu {num_nos_sugeridos} nós com base nas marcações amarelas.")

    with col2:
        st.subheader("⚙️ Validação e Ajuste de Nós/Barras")
        st.warning("Se o OpenCV identificou barras ou nós fantasmas devido às setas, corrija os valores abaixo:")
        
        # CONTROLE MANUAL QUE SUBSTITUI O ERRO DO OPENCV
        num_nos_reais = st.number_input(
            "Quantidade REAL de nós da estrutura (Ex: Pórtico simples = 4 nós):", 
            min_value=2, 
            max_value=20, 
            value=int(num_nos_sugeridos) if num_nos_sugeridos <= 4 else 4,
            step=1
        )
        
        # Número de barras em estruturas aporticadas lineares simples é igual a nós - 1
        num_barras_reais = num_nos_reais - 1
        st.success(f"Configurado para calcular: {num_nos_reais} Nós e {num_barras_reais} Barra(s).")
        
        st.markdown("---")
        st.markdown("#### Dimensões das Barras (m)")
        
        comprimentos = {}
        # Se for o pórtico clássico de 4 nós, nomeia os campos para facilitar a engenharia
        if num_nos_reais == 4:
            comprimentos[0] = st.number_input("Altura do Pilar Esquerdo (Barra 1) [m]:", value=4.0, min_value=0.1)
            comprimentos[1] = st.number_input("Comprimento da Viga Superior (Barra 2) [m]:", value=6.0, min_value=0.1)
            comprimentos[2] = st.number_input("Altura do Pilar Direito (Barra 3) [m]:", value=4.0, min_value=0.1)
        else:
            # Caso geral se você mudar para vigas contínuas ou outros formatos
            for i in range(num_barras_reais):
                comprimentos[i] = st.number_input(f"Comprimento da Barra {i+1} [m]:", value=4.0, min_value=0.1, key=f"c_manual_{i}")

        st.markdown("---")
        st.markdown("#### Configuração de Cargas")
        
        no_carga = st.selectbox("Selecione o Nó da Força Concentrada (Verde):", [f"Nó {i+1}" for i in range(num_nos_reais)], index=1)
        val_fx = st.number_input(f"Força Horizontal Fx no {no_carga} (kN) [→ é positivo]:", value=10.0)
        val_fy = st.number_input(f"Força Vertical Fy no {no_carga} (kN) [↓ é negativo]:", value=0.0)
        
        st.markdown("---")
        tem_distribuida = st.checkbox("Existe carga distribuída na viga?", value=True)
        val_q = 0.0
        if tem_distribuida:
            val_q = st.number_input("Intensidade da Carga Distribuída (kN/m) [↓ é negativo]:", value=-10.0)

    # --- PROCESSAMENTO MATRICIAL (ANASTUCT) ---
    st.markdown("---")
    if st.button("🚀 Resolver Estrutura via Rigidez Direta"):
        try:
            ss = SystemElements()
            
            # Construção geométrica baseada no número real validado pelo usuário
            if num_nos_reais == 4:
                h1 = comprimentos[0]
                L_viga = comprimentos[1]
                h2 = comprimentos[2]
                
                # Coordenadas locais do pórtico (X, Y)
                coords = {
                    1: [0.0, 0.0],       # Base pilar esquerdo
                    2: [0.0, h1],        # Topo pilar esquerdo
                    3: [L_viga, h1],     # Topo pilar direito
                    4: [L_viga, h1 - h2] # Base pilar direito
                }
                
                # Adiciona pilares e viga
                ss.add_element(location=[coords[1], coords[2]], EA=E_val*A_val, EI=E_val*I_val)
                ss.add_element(location=[coords[2], coords[3]], EA=E_val*A_val, EI=E_val*I_val)
                ss.add_element(location=[coords[3], coords[4]], EA=E_val*A_val, EI=E_val*I_val)
                
                # Condições de Contorno (Apoios conforme seu desenho)
                ss.add_support_fixed(node_id=1) # Engaste na base esquerda
                ss.add_support_fixed(node_id=4) # Engaste na base direita
                
                # Carga distribuída atua na viga superior (elemento 2)
                if tem_distribuida and val_q != 0:
                    ss.add_distributed_load(element_id=2, q=val_q)
                    
            else:
                # Fallback genérico de viga contínua reta se você mudar o número de nós
                coords = {1: [0.0, 0.0]}
                for i in range(num_barras_reais):
                    coords[i+2] = [coords[i+1][0] + comprimentos[i], 0.0]
                    ss.add_element(location=[coords[i+1], coords[i+2]], EA=E_val*A_val, EI=E_val*I_val)
                ss.add_support_fixed(node_id=1)
                ss.add_support_hinged(node_id=num_nos_reais)

            # Aplicação da carga pontual parametrizada com tratamento de redundância de versão
            idx_no_carga = int(no_carga.split(" ")[1])
            if val_fx != 0 or val_fy != 0:
                try:
                    ss.add_node_q(node_id=idx_no_carga, Fx=val_fx, Fy=val_fy)
                except AttributeError:
                    try:
                        ss.add_point_load(node_id=idx_no_carga, Fx=val_fx, Fy=val_fy)
                    except AttributeError:
                        ss.add_node_load(node_id=idx_no_carga, Fx=val_fx, Fy=val_fy)

            # Execução do Motor MRD
            ss.solve()

            # --- PLOTAGEM DOS DIAGRAMAS ---
            st.subheader("📊 Diagramas de Esforços Solicitantes Obtidos")
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
                st.write("**Esforço Axial / Normal (N)**")
                fig_n, ax_n = plt.subplots()
                ss.plot_axial_force(show=False, ax=ax_n)
                st.pyplot(fig_n)
                
            st.success("🎯 Deslocamentos dos Nós Calculados (Vetor U):")
            for node in ss.get_node_displacements():
                st.write(f"**Nó {node['id']}:** Ux = {node['ux']:.4f} m | Uy = {node['uy']:.4f} m | Rotação θz = {node['phi']:.4f} rad")

        except Exception as erro:
            st.error(f"Erro na análise matricial da estrutura: {erro}")
            st.info("Dica: Certifique-se de que os comprimentos informados condizem com as restrições da estrutura.")
else:
    st.warning("Insira a foto para iniciar o personalizador.")
