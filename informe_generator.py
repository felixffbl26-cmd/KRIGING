"""
Módulo de generación de informes profesionales en formato A4
para el sistema de Kriging Geoestadístico
"""

def generar_informe_html(res, df_safe, economia_data, project_name, student_names):
    """
    Genera un informe HTML profesional en formato A4 con todas las secciones
    
    Args:
        res: Diccionario con resultados de kriging
        df_safe: DataFrame con datos de muestras
        economia_data: Diccionario con datos económicos (puede ser None)
        project_name: Nombre del proyecto
        student_names: Lista de nombres de estudiantes
    
    Returns:
        str: HTML del informe completo
    """
    from datetime import datetime
    
    # Generar lista HTML de estudiantes
    est_li = "".join([f"<li>{e}</li>" for e in student_names])
    
    # Detectar columnas
    col_id = next((c for c in ['ID', 'Id', 'id', 'Muestra'] if c in df_safe.columns), df_safe.columns[0])
    col_ley = next((c for c in ['Ley', 'LEY', 'ley', 'Grade'] if c in df_safe.columns), df_safe.columns[-1])
    
    # Generar tabla de muestras (máximo 50)
    rows = ""
    for i in range(min(len(df_safe), 50)):
        try:
            val_id = str(df_safe[col_id].iloc[i])
            val_ley = float(df_safe[col_ley].iloc[i])
            val_dist = float(res['d_vec'][i])
            val_peso = float(res['pesos'][i])
            
            rows += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 6px; font-size: 0.85em;">{val_id}</td>
                <td style="border: 1px solid #ddd; padding: 6px; text-align: right; font-size: 0.85em;">{val_dist:.2f}</td>
                <td style="border: 1px solid #ddd; padding: 6px; text-align: right; font-size: 0.85em;">{val_peso:.4f}</td>
                <td style="border: 1px solid #ddd; padding: 6px; text-align: right; font-size: 0.85em;">{val_ley:.2f}</td>
            </tr>"""
        except:
            continue
    
    # Sección de economía
    if economia_data:
        economia_html = f"""
        <div style="page-break-before: always;"></div>
        <h3 style="color:#0277bd; border-bottom:2px solid #0277bd; padding-bottom:5px; margin-top:30px;">3. VALORIZACIÓN ECONÓMICA</h3>
        
        <div style="background-color:#fff3cd; padding:20px; border-radius:8px; border-left: 5px solid #ffc107; margin: 20px 0;">
            <h4 style="color:#856404; margin-top:0;">💰 Evaluación Económica del Bloque</h4>
            <p style="margin:5px 0; color:#333;">
                El análisis económico determina la viabilidad de extracción del bloque considerando 
                costos operativos, recuperación metalúrgica y precios de mercado.
            </p>
        </div>
        
        <table style="width:100%; border-collapse: collapse; margin: 20px 0;">
            <tr style="background-color:#0277bd; color:white;">
                <th style="padding: 10px; text-align: left;" colspan="2">PARÁMETROS GEOMÉTRICOS Y FÍSICOS</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px; width: 50%;"><b>Volumen del Bloque:</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{economia_data['volumen']:,.0f} m³</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><b>Tonelaje Total:</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{economia_data['tonelaje']:,.0f} toneladas</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><b>Ley Diluida:</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{economia_data['ley_diluida']:.3f} %</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><b>Metal Recuperable:</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{economia_data['metal_recuperable']:.2f} toneladas finas</td>
            </tr>
        </table>
        
        <table style="width:100%; border-collapse: collapse; margin: 20px 0;">
            <tr style="background-color:#0277bd; color:white;">
                <th style="padding: 10px; text-align: left;" colspan="2">ANÁLISIS FINANCIERO</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px; width: 50%;"><b>Ingresos Brutos (NSR):</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">US$ {economia_data['ingreso_bruto']:,.2f}</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><b>Costos Totales:</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">US$ {economia_data['costo_total']:,.2f}</td>
            </tr>
            <tr style="background-color: {'#d4edda' if economia_data['beneficio'] > 0 else '#f8d7da'};">
                <td style="border: 1px solid #ddd; padding: 8px;"><b>BENEFICIO NETO:</b></td>
                <td style="border: 1px solid #ddd; padding: 8px; font-size: 1.2em; font-weight: bold; color: {'#155724' if economia_data['beneficio'] > 0 else '#721c24'};">
                    US$ {economia_data['beneficio']:,.2f}
                </td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><b>Margen de Beneficio:</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{economia_data['margen']:.1f} %</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><b>Ley de Corte Calculada:</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{economia_data['ley_corte']:.3f} %</td>
            </tr>
        </table>
        
        <div style="background-color: {'#d4edda' if economia_data['beneficio'] > 0 else '#f8d7da'}; 
                    padding:20px; border-radius:8px; 
                    border: 3px solid {'#28a745' if economia_data['beneficio'] > 0 else '#dc3545'}; 
                    text-align:center; margin: 20px 0;">
            <h3 style="color: {'#155724' if economia_data['beneficio'] > 0 else '#721c24'}; margin:0;">
                {economia_data['decision']}
            </h3>
            <p style="margin:10px 0; color:#666;">
                Ley Estimada ({economia_data['ley_diluida']:.3f}%) 
                {'>' if economia_data['beneficio'] > 0 else '<'} 
                Ley de Corte ({economia_data['ley_corte']:.3f}%)
            </p>
        </div>
        """
    else:
        economia_html = """
        <div style="page-break-before: always;"></div>
        <h3 style="color:#0277bd; border-bottom:2px solid #0277bd; padding-bottom:5px;">3. VALORIZACIÓN ECONÓMICA</h3>
        <p style="color:#666; font-style:italic;">No se ha realizado evaluación económica. Complete la Pestaña 5 para generar esta sección.</p>
        """
    
    # HTML completo
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{ size: A4; margin: 2cm; }}
            @media print {{ .no-print {{ display: none; }} body {{ margin: 0; }} }}
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                background: white;
                color: #333;
                line-height: 1.6;
                max-width: 21cm;
                margin: 0 auto;
                padding: 20px;
            }}
        </style>
    </head>
    <body>
    
    <!-- PORTADA -->
    <div style="text-align:center; border: 3px solid #0277bd; padding:40px; margin-bottom: 30px; background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%);">
        <h1 style="color:#0277bd; margin:0; font-size: 2.2em; text-transform: uppercase;">INFORME TÉCNICO</h1>
        <h2 style="color:#01579b; margin:10px 0; font-size: 1.5em;">Estimación de Recursos Minerales</h2>
        <h3 style="color:#0277bd; margin:15px 0; font-size: 1.3em; font-weight: normal;">Método: Kriging Ordinario</h3>
        <hr style="border: 1px solid #0277bd; width: 60%; margin: 20px auto;">
        <h2 style="color:#333; margin:20px 0; font-size: 1.4em;">{project_name.upper()}</h2>
    </div>
    
    <!-- INFORMACIÓN DEL PROYECTO -->
    <table style="width:100%; margin-bottom: 30px; border-collapse: collapse;">
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd; background-color: #f5f5f5; width: 30%;"><b>Docente Supervisor:</b></td>
            <td style="padding: 10px; border: 1px solid #ddd;">Ing. Arturo R. Chayña Rodriguez</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd; background-color: #f5f5f5;"><b>Fecha de Emisión:</b></td>
            <td style="padding: 10px; border: 1px solid #ddd;">{datetime.now().strftime('%d de %B de %Y')}</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd; background-color: #f5f5f5;"><b>Equipo Técnico:</b></td>
            <td style="padding: 10px; border: 1px solid #ddd;">
                <ul style="margin:5px 0; padding-left:20px;">{est_li}</ul>
            </td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd; background-color: #f5f5f5;"><b>Número de Muestras:</b></td>
            <td style="padding: 10px; border: 1px solid #ddd;">{len(df_safe)} muestras procesadas</td>
        </tr>
    </table>
    
    <!-- RESUMEN EJECUTIVO -->
    <h3 style="color:#0277bd; border-bottom:2px solid #0277bd; padding-bottom:5px;">1. RESUMEN EJECUTIVO</h3>
    <div style="background-color:#e1f5fe; padding:20px; border-radius:8px; border-left: 5px solid #0277bd; margin: 20px 0;">
        <p style="margin:0; font-size: 1.05em;">
            Se ha realizado la estimación geoestadística del bloque centrado en las coordenadas 
            <b>Este: {res['tx']:.2f} m</b> y <b>Norte: {res['ty']:.2f} m</b> utilizando el método de 
            <b>Kriging Ordinario</b> con modelo variográfico esférico.
        </p>
        
        <table style="width:100%; margin-top: 15px; border-collapse: collapse;">
            <tr style="background-color:#0277bd; color:white;">
                <th style="padding: 8px; text-align: left;">Parámetro</th>
                <th style="padding: 8px; text-align: left;">Valor</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><b>LEY ESTIMADA (Z*):</b></td>
                <td style="border: 1px solid #ddd; padding: 8px; font-size: 1.2em; font-weight: bold; color: #0277bd;">{res['ley']:.4f} %</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><b>Varianza de Kriging (σ²):</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{res['var']:.4f}</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><b>Desviación Estándar (σ):</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{res['sigma']:.4f}</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><b>Coeficiente de Variación:</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{res['cv_k']:.2f} %</td>
            </tr>
            <tr style="background-color: {'#d4edda' if res['cat'] == 'MEDIDO' else '#fff3cd' if res['cat'] == 'INDICADO' else '#f8d7da'};">
                <td style="border: 1px solid #ddd; padding: 8px;"><b>CLASIFICACIÓN JORC:</b></td>
                <td style="border: 1px solid #ddd; padding: 8px; font-size: 1.1em; font-weight: bold; color: {'#155724' if res['cat'] == 'MEDIDO' else '#856404' if res['cat'] == 'INDICADO' else '#721c24'};">
                    {res['cat']}
                </td>
            </tr>
        </table>
    </div>
    
    <!-- METODOLOGÍA -->
    <div style="page-break-before: always;"></div>
    <h3 style="color:#0277bd; border-bottom:2px solid #0277bd; padding-bottom:5px; margin-top:30px;">2. METODOLOGÍA GEOESTADÍSTICA</h3>
    
    <h4 style="color:#01579b; margin-top:20px;">2.1. Modelo Variográfico</h4>
    <p>Se utilizó un modelo variográfico <b>Esférico</b> con los siguientes parámetros:</p>
    
    <table style="width:100%; border-collapse: collapse; margin: 15px 0;">
        <tr style="background-color:#0277bd; color:white;">
            <th style="padding: 8px; text-align: left;">Parámetro</th>
            <th style="padding: 8px; text-align: left;">Símbolo</th>
            <th style="padding: 8px; text-align: left;">Valor</th>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Efecto Pepita (Nugget)</td>
            <td style="border: 1px solid #ddd; padding: 8px;">C₀</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{res['c0']:.3f}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Meseta Parcial</td>
            <td style="border: 1px solid #ddd; padding: 8px;">C₁</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{res['c1']:.3f}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Rango (Alcance)</td>
            <td style="border: 1px solid #ddd; padding: 8px;">a</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{res['a']:.1f} metros</td>
        </tr>
        <tr style="background-color:#f5f5f5;">
            <td style="border: 1px solid #ddd; padding: 8px;"><b>Meseta Total (Sill)</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>C₀ + C₁</b></td>
            <td style="border: 1px solid #ddd; padding: 8px;"><b>{res['c0'] + res['c1']:.3f}</b></td>
        </tr>
    </table>
    
    <h4 style="color:#01579b; margin-top:20px;">2.2. Sistema de Kriging Ordinario</h4>
    <p>El sistema de ecuaciones resuelto fue:</p>
    <div style="background-color:#f5f5f5; padding:15px; border-radius:5px; margin:10px 0; text-align:center;">
        <p style="font-family: 'Courier New', monospace; font-size: 1.1em; margin:0;">[K] · [W] = [M]</p>
    </div>
    <p style="font-size:0.95em; color:#666;">
        Donde [K] es la matriz de covarianzas entre muestras, [W] el vector de pesos incógnita, 
        y [M] el vector de covarianzas muestra-bloque. El sistema incluye la restricción de insesgo 
        mediante el multiplicador de Lagrange (μ = {res['mu']:.4f}).
    </p>
    
    {economia_html}
    
    <!-- CLASIFICACIÓN JORC -->
    <div style="page-break-before: always;"></div>
    <h3 style="color:#0277bd; border-bottom:2px solid #0277bd; padding-bottom:5px; margin-top:30px;">4. CLASIFICACIÓN DE RECURSOS (JORC/NI 43-101)</h3>
    
    <div style="background-color:#fff3cd; padding:20px; border-radius:8px; border-left: 5px solid #ffc107; margin: 20px 0;">
        <h4 style="color:#856404; margin-top:0;">⚖️ Criterios de Clasificación</h4>
        <p style="margin:5px 0; color:#333;">
            La clasificación se basa en el <b>Coeficiente de Variación del Kriging (CV)</b>, 
            que mide la incertidumbre relativa de la estimación.
        </p>
    </div>
    
    <table style="width:100%; border-collapse: collapse; margin: 20px 0;">
        <tr style="background-color:#0277bd; color:white;">
            <th style="padding: 10px;">Categoría</th>
            <th style="padding: 10px;">CV Kriging (%)</th>
            <th style="padding: 10px;">Nivel de Confianza</th>
        </tr>
        <tr style="background-color: {'#d4edda' if res['cat'] == 'MEDIDO' else '#f5f5f5'};">
            <td style="border: 1px solid #ddd; padding: 8px; text-align:center;"><b>MEDIDO</b></td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align:center;">< 15%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Alta confianza - Reservas Probadas</td>
        </tr>
        <tr style="background-color: {'#fff3cd' if res['cat'] == 'INDICADO' else '#f5f5f5'};">
            <td style="border: 1px solid #ddd; padding: 8px; text-align:center;"><b>INDICADO</b></td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align:center;">15% - 30%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Confianza razonable - Reservas Probables</td>
        </tr>
        <tr style="background-color: {'#f8d7da' if res['cat'] == 'INFERIDO' else '#f5f5f5'};">
            <td style="border: 1px solid #ddd; padding: 8px; text-align:center;"><b>INFERIDO</b></td>
            <td style="border: 1px solid #ddd; padding: 8px; text-align:center;">> 30%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Baja confianza - Solo exploración</td>
        </tr>
    </table>
    
    <div style="background-color: {'#d4edda' if res['cat'] == 'MEDIDO' else '#fff3cd' if res['cat'] == 'INDICADO' else '#f8d7da'}; 
                padding:25px; border-radius:10px; 
                border: 3px solid {'#28a745' if res['cat'] == 'MEDIDO' else '#ffc107' if res['cat'] == 'INDICADO' else '#dc3545'}; 
                text-align:center; margin: 25px 0;">
        <h3 style="color: {'#155724' if res['cat'] == 'MEDIDO' else '#856404' if res['cat'] == 'INDICADO' else '#721c24'}; margin:0; font-size:1.8em;">
            RECURSO {res['cat']}
        </h3>
        <p style="margin:15px 0; font-size:1.2em; color:#333;">
            CV = {res['cv_k']:.2f}%
        </p>
    </div>
    
    <!-- MEMORIA DE CÁLCULO -->
    <div style="page-break-before: always;"></div>
    <h3 style="color:#0277bd; border-bottom:2px solid #0277bd; padding-bottom:5px; margin-top:30px;">5. MEMORIA DE CÁLCULO</h3>
    <p style="font-size:0.95em; color:#666;">
        Se procesaron un total de <b>{len(df_safe)} muestras</b>. 
        A continuación se muestra el detalle de las primeras 50 muestras:
    </p>
    
    <table style="width:100%; border-collapse: collapse; font-size: 0.85em; margin-top:15px;">
        <tr style="background-color:#0277bd; color:white;">
            <th style="padding: 8px;">ID</th>
            <th style="padding: 8px; text-align:right;">Dist (m)</th>
            <th style="padding: 8px; text-align:right;">Peso (λ)</th>
            <th style="padding: 8px; text-align:right;">Ley (%)</th>
        </tr>
        {rows}
    </table>
    
    {f'<p style="font-size:0.85em; color:#666; margin-top:10px;"><i>Nota: Se muestran las primeras 50 de {len(df_safe)} muestras totales.</i></p>' if len(df_safe) > 50 else ''}
    
    <!-- FIRMAS -->
    <div style="margin-top:80px; page-break-inside: avoid;">
        <table style="width:100%; text-align:center;">
            <tr>
                <td style="width:50%; padding:20px;">
                    <div style="border-top:2px solid #333; padding-top:10px; margin:0 40px;">
                        <p style="margin:5px 0;"><b>Ing. Arturo R. Chayña Rodriguez</b></p>
                        <p style="margin:5px 0; font-size:0.9em; color:#666;">Docente Supervisor</p>
                    </div>
                </td>
                <td style="width:50%; padding:20px;">
                    <div style="border-top:2px solid #333; padding-top:10px; margin:0 40px;">
                        <p style="margin:5px 0;"><b>Equipo Técnico</b></p>
                        <p style="margin:5px 0; font-size:0.9em; color:#666;">Estudiantes</p>
                    </div>
                </td>
            </tr>
        </table>
    </div>
    
    <!-- PIE DE PÁGINA -->
    <div style="margin-top:40px; padding-top:20px; border-top:2px solid #0277bd; text-align:center; color:#777; font-size:0.85em;">
        <p style="margin:5px 0;">Universidad Nacional del Altiplano - Puno</p>
        <p style="margin:5px 0;">Facultad de Ingeniería de Minas</p>
        <p style="margin:5px 0;">Generado por Sistema Geo-Miner Pro v2.0 | {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
    </div>
    
    </body>
    </html>
    """
    
    return html
